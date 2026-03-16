#include <string.h>
#include <stddef.h>
#define _GNU_SOURCE
#include <time.h>
#include <sys/sysinfo.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "tflite/c/c_api.h"
#include "tflite/c/c_api_experimental.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "inc/util.h"
#include "json_util.h"
#include "utlist.h"
#include "./cable_autoencoder_xnnpack.h"

const char *states_string[] = {"normal", "suspicious", "anomalous"};
const char *const feature_names[14] = {
    "frame_err_ppm",
    "length_err_ppm",
    "speed_change_count_10m",
    "speed_is_downgraded",
    "rx_err_ppm",
    "phy_receive_errors_rate",
    "phy_serdes_ber_errors_rate",
    "fcs_per_million_pkts",
    "rx_error_rate",
    "host_rx_crc_rate",
    "tx_dropped_rate",
    "utilization",
    "flaps_10m",
    "temp_slope_10m"
};

/* GLOBALS */
struct model_config config;
cable_features current_features;
cable_features prev_features;

static float nonneg_delta_u64(uint64_t curr, uint64_t prev) {
    if (curr < prev) return NAN;
    return (float)(curr - prev);
}

static float safe_rate_u64(uint64_t curr, uint64_t prev, double dt_sec) {
    if (dt_sec <= 0.0 || curr < prev) return NAN;
    return (float)((curr - prev) / dt_sec);
}

static void rolling_push(rolling_window *w, double ts, float value, double span_sec) {
    size_t end;

    if (w->count == WINDOW_CAP) {
        w->start = (w->start + 1) % WINDOW_CAP;
        w->count--;
    }

    end = (w->start + w->count) % WINDOW_CAP;
    w->ts[end] = ts;
    w->val[end] = value;
    w->count++;

    while (w->count > 0 && (ts - w->ts[w->start]) > span_sec) {
        w->start = (w->start + 1) % WINDOW_CAP;
        w->count--;
    }
}

static float rolling_sum(const rolling_window *w) {
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx])) sum += w->val[idx];
    }
    return sum;
}

static float rolling_first(const rolling_window *w) {
    size_t i;
    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx])) return w->val[idx];
    }
    return NAN;
}

static int parse_ethtool_kv_u64(uint64_t *stat, char *ethtool_output, const char *input)
{
    char *line;
    char *saveptr;

    if (!stat || !ethtool_output || !input) {
        return -1;
    }

    line = strtok_r(ethtool_output, "\n", &saveptr);
    while (line != NULL) {
        char *colon = strchr(line, ':');
        if (colon != NULL) {
            char *key_start = line;
            char *key_end = colon - 1;
            char *val_start = colon + 1;
            unsigned long long value;

            while (*key_start == ' ' || *key_start == '\t') {
                key_start++;
            }
            while (key_end >= key_start && (*key_end == ' ' || *key_end == '\t')) {
                *key_end = '\0';
                key_end--;
            }
            while (*val_start == ' ' || *val_start == '\t') {
                val_start++;
            }

            if (strcmp(key_start, input) == 0) {
                value = strtoull(val_start, NULL, 10);
                *stat = (uint64_t)value;
                return 0;
            }
        }

        line = strtok_r(NULL, "\n", &saveptr);
    }

    return -1;
}

static int compute_cable_features( cable_feature_state *state,
                                   const raw_sample *curr,
                                   cable_features *out,
                                   cable_history_metrics* out_history_metrics) {
    double dt_sec;
    float rx_pkts_delta;
    float frame_delta;
    float len_delta;
    float rx_err_delta;
    float fcs_delta;
    float rx_bps;
    float tx_bps;
    float temp_first;
    float speed_changed = 0.0f;
    float speed_downgraded = 0.0f;
    float link_flap = 0.0f;

    memset(out, 0, sizeof(*out));
    memset(out_history_metrics, 0, sizeof(*out_history_metrics));

    if (!state->initialized) {
        state->prev = *curr;
        state->initialized = 1;
        return 1;
    }

    dt_sec = curr->ts_sec - state->prev.ts_sec;
    if (dt_sec <= 0.0) {
        return -1;
    }

    if (curr->speed_mbps != state->prev.speed_mbps) {
        speed_changed = 1.0f;
        if (curr->speed_mbps < state->prev.speed_mbps) {
            speed_downgraded = 1.0f;
        }
    }

    if (curr->carrier != state->prev.carrier) {
        link_flap = 1.0f;
    }

    rolling_push(&state->speed_change_10m, curr->ts_sec, speed_changed, SPAN_10MIN);
    rolling_push(&state->link_flap_10m, curr->ts_sec, link_flap, SPAN_10MIN);
    rolling_push(&state->temp_10m, curr->ts_sec, curr->temp_c, SPAN_10MIN);
    rolling_push(&state->speed_change_1h, curr->ts_sec, speed_changed, SPAN_1H);
    rolling_push(&state->link_flap_1h, curr->ts_sec, link_flap, SPAN_1H);
    rolling_push(&state->temp_1h, curr->ts_sec, curr->temp_c, SPAN_1H);

    rx_pkts_delta = nonneg_delta_u64(curr->rx_packets, state->prev.rx_packets);
    frame_delta = nonneg_delta_u64(curr->rx_frame_errors, state->prev.rx_frame_errors);
    len_delta = nonneg_delta_u64(curr->rx_length_errors, state->prev.rx_length_errors);
    rx_err_delta = nonneg_delta_u64(curr->rx_errors, state->prev.rx_errors);
    fcs_delta = nonneg_delta_u64(curr->rx_crc_errors, state->prev.rx_crc_errors);

    out->frame_err_ppm = (!isnan(frame_delta) && !isnan(rx_pkts_delta)) ? (1000000.0f * frame_delta / fmaxf(rx_pkts_delta, 1.0f))
                                                                        : NAN;

    out->length_err_ppm = (!isnan(len_delta) && !isnan(rx_pkts_delta)) ? (1000000.0f * len_delta / fmaxf(rx_pkts_delta, 1.0f))
                                                                       : NAN;

    out->speed_change_count_10m = rolling_sum(&state->speed_change_10m);
    out->speed_is_downgraded = speed_downgraded;

    out->rx_err_ppm = (!isnan(rx_err_delta) && !isnan(rx_pkts_delta)) ? (1000000.0f * rx_err_delta / fmaxf(rx_pkts_delta, 1.0f))
                                                                      : NAN;

    out->phy_receive_errors_rate = safe_rate_u64(curr->phy_receive_errors, state->prev.phy_receive_errors, dt_sec);
    out->phy_serdes_ber_errors_rate = safe_rate_u64(curr->phy_serdes_ber_errors, state->prev.phy_serdes_ber_errors, dt_sec);
    out->fcs_per_million_pkts = (!isnan(fcs_delta) && !isnan(rx_pkts_delta)) ? (1000000.0f * fcs_delta / fmaxf(rx_pkts_delta, 1.0f))
                                                                             : NAN;

    out->rx_error_rate = safe_rate_u64(curr->rx_errors, state->prev.rx_errors, dt_sec);
    out->host_rx_crc_rate = safe_rate_u64(curr->rx_crc_errors, state->prev.rx_crc_errors, dt_sec);
    out->tx_dropped_rate = safe_rate_u64(curr->tx_dropped, state->prev.tx_dropped, dt_sec);

    rx_bps = safe_rate_u64(curr->rx_bytes, state->prev.rx_bytes, dt_sec) * BITS_PER_BYTE;
    tx_bps = safe_rate_u64(curr->tx_bytes, state->prev.tx_bytes, dt_sec) * BITS_PER_BYTE;

    out->utilization = (curr->speed_mbps > 0 && !isnan(rx_bps) && !isnan(tx_bps)) ? ((rx_bps + tx_bps) / ((float)curr->speed_mbps * 1000000.0f))
                                                                                  : NAN;

    out->flaps_10m = rolling_sum(&state->link_flap_10m);

    temp_first = rolling_first(&state->temp_10m);
    out->temp_slope_10m = (!isnan(temp_first) && !isnan(curr->temp_c)) ? ((curr->temp_c - temp_first) / 600.0f)
                                                                       : NAN;

    /* cable history metrics */
    out_history_metrics->speed_change_count_1h = rolling_sum(&state->speed_change_1h);
    out_history_metrics->flaps_1h = rolling_sum(&state->link_flap_1h);
    float first_temp_1h = rolling_first(&state->temp_1h);
    out_history_metrics->temp_slope_1h = (!isnan(curr->temp_c) && !isnan(first_temp_1h)) ? (curr->temp_c - first_temp_1h) / 3600.0f
                                                                                         : NAN;

    state->prev = *curr;
    return 0;
}

static void pack_model_input(const cable_features *f, float input[14]) {
    input[0]  = f->frame_err_ppm;
    input[1]  = f->length_err_ppm;
    input[2]  = f->speed_change_count_10m;
    input[3]  = f->speed_is_downgraded;
    input[4]  = f->rx_err_ppm;
    input[5]  = f->phy_receive_errors_rate;
    input[6]  = f->phy_serdes_ber_errors_rate;
    input[7]  = f->fcs_per_million_pkts;
    input[8]  = f->rx_error_rate;
    input[9]  = f->host_rx_crc_rate;
    input[10] = f->tx_dropped_rate;
    input[11] = f->utilization;
    input[12] = f->flaps_10m;
    input[13] = f->temp_slope_10m;
}

void model_inputs_outputs(TfLiteInterpreter* interpreter)
{
    printf("=== Inputs ===\n");
    int in_count = TfLiteInterpreterGetInputTensorCount(interpreter);
    for (int i = 0; i < in_count; ++i) {
        const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interpreter, i);
        printf("in[%d] name=%s type=%d bytes=%zu dims=%d [", i, TfLiteTensorName(t),
            TfLiteTensorType(t), TfLiteTensorByteSize(t), TfLiteTensorNumDims(t));
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
            printf("%d%s", TfLiteTensorDim(t, d), d + 1 == TfLiteTensorNumDims(t) ? "" : ",");
        }
        printf("]\n");
    }

    printf("=== Outputs ===\n");
    int out_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
    for (int i = 0; i < out_count; ++i) {
        const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interpreter, i);
        printf("out[%d] name=%s type=%d bytes=%zu dims=%d [", i, TfLiteTensorName(t),
            TfLiteTensorType(t), TfLiteTensorByteSize(t), TfLiteTensorNumDims(t));
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
            printf("%d%s", TfLiteTensorDim(t, d), d + 1 == TfLiteTensorNumDims(t) ? "" : ",");
        }
        printf("]\n");
    }
}

static inline int64_t timespec_diff_ns( struct timespec end, struct timespec start)
{
    return (int64_t)(end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
}

static int invoke_model( TfLiteInterpreter *interpreter,
                         cable_feature_state *feature_state,
                         const raw_sample *sample,
                         cable_features *features,
                         cable_history_metrics *history_metrics,
                         float *reconstruction_error,
                         float *reconstruction_error_per_feature) {
    float model_input[NUM_FEATURES];
    float model_output[NUM_FEATURES];
    TfLiteTensor *in;
    const TfLiteTensor *out;
    int feat_rc;
    int i;
    struct timespec start, end;
    float mse = 0.0f;

    /* calculate cable features from samples, update feature state */
    feat_rc = compute_cable_features(feature_state, sample, features, history_metrics);
    if (feat_rc != 0) {
        return feat_rc;
    }

    pack_model_input(features, model_input);

    in = TfLiteInterpreterGetInputTensor(interpreter, 0);
    out = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    if (!in || !out) {
        fprintf(stderr, "tensor lookup failed\n");
        return -1;
    }

    if (TfLiteTensorCopyFromBuffer(in, model_input, sizeof(model_input)) != kTfLiteOk) {
        fprintf(stderr, "input copy failed\n");
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "invoke failed\n");
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int64_t elapsed_ns = timespec_diff_ns(end, start);
    printf("Inference duration time: %ld [us]\n", elapsed_ns/1000);

    if (TfLiteTensorCopyToBuffer(out, model_output, sizeof(model_output)) != kTfLiteOk) {
        fprintf(stderr, "output copy failed\n");
        return -1;
    }

    for (i = 0; i < NUM_FEATURES; ++i) {
        float d = model_input[i] - model_output[i];
        reconstruction_error_per_feature[i] = fabsf(d);
        mse += d * d;
    }
    mse /= (float)NUM_FEATURES;

    if (reconstruction_error) {
        *reconstruction_error = mse;
    }

    return 0;
}

static uint64_t read_u64_stat(const char *iface, const char *name)
{
    char path[256];
    FILE *fp;
    unsigned long long value = 0;

    snprintf(path, sizeof(path), "/sys/class/net/%s/statistics/%s", iface, name);

    fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }

    if (fscanf(fp, "%llu", &value) != 1) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return (uint64_t)value;
}

static int read_u64_stat_or_default(const char *iface, const char *name, uint64_t *value)
{
    char path[256];
    FILE *fp;
    unsigned long long tmp = 0;

    if (!value) {
        return -1;
    }

    *value = 0;

    snprintf(path, sizeof(path), "/sys/class/net/%s/statistics/%s", iface, name);

    fp = fopen(path, "r");
    if (!fp) {
        return -1;
    }

    if (fscanf(fp, "%llu", &tmp) != 1) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *value = (uint64_t)tmp;
    return 0;
}

static int read_int_sys(const char *iface, const char *name)
{
    char path[256];
    FILE *fp;
    int value = 0;

    snprintf(path, sizeof(path), "/sys/class/net/%s/%s", iface, name);

    fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }

    if (fscanf(fp, "%d", &value) != 1) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return value;
}

static int run_command(const char *command, char *output, size_t output_size)
{
    FILE *fp;
    char line[512];

    if (!command || !output || output_size == 0) {
        return -1;
    }

    fp = popen(command, "r");
    if (!fp) {
        return -1;
    }

    output[0] = '\0';

    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t used = strlen(output);
        size_t left;

        if (used >= output_size - 1) {
            break;
        }

        left = output_size - used - 1;
        strncat(output, line, left);
    }

    if (pclose(fp) == -1) {
        return -1;
    }

    return 0;
}

static int ethtool_phy_stats(const char *iface, struct interface_stats *i_stats)
{
    char command[128];
    char output[8192];
    char output1[8192];
    char output2[8192];

    memset(command, 0, sizeof(command));
    memset(output, 0, sizeof(output));
    memset(output1, 0, sizeof(output1));
    memset(output2, 0, sizeof(output2));

    snprintf(command, sizeof(command), "ethtool --phy-statistics %s", iface);

    if (run_command(command, output, sizeof(output)) != 0) {
        return -1;
    }

    strncpy(output1, output, sizeof(output1) - 1);
    strncpy(output2, output, sizeof(output2) - 1);

    parse_ethtool_kv_u64(&i_stats->phy_receive_errors, output1, "phy_receive_errors");
    parse_ethtool_kv_u64(&i_stats->phy_serdes_ber_errors, output2, "phy_serdes_ber_errors");

    return 0;
}

static int ethtool_port_stats(const char *iface, struct interface_stats *i_stats)
{
    char command[128];
    char output[8192];
    char output1[8192];
    char output2[8192];
    char output3[8192];
    char output4[8192];

    memset(command, 0, sizeof(command));
    memset(output, 0, sizeof(output));
    memset(output1, 0, sizeof(output1));
    memset(output2, 0, sizeof(output2));
    memset(output3, 0, sizeof(output3));
    memset(output4, 0, sizeof(output4));

    snprintf(command, sizeof(command), "ethtool -S %s", iface);

    if (run_command(command, output, sizeof(output)) != 0) {
        return -1;
    }

    strncpy(output1, output, sizeof(output1) - 1);
    strncpy(output2, output, sizeof(output2) - 1);
    strncpy(output3, output, sizeof(output3) - 1);
    strncpy(output4, output, sizeof(output4) - 1);

    i_stats->rx_frame_errors = 0;
    i_stats->rx_length_errors = 0;

    if (parse_ethtool_kv_u64(&i_stats->rx_frame_errors, output1, "rx_frame_errors") != 0) {
        parse_ethtool_kv_u64(&i_stats->rx_frame_errors, output2, "rx_alignment_errors");
    }

    if (parse_ethtool_kv_u64(&i_stats->rx_length_errors, output3, "rx_length_errors") != 0) {
        parse_ethtool_kv_u64(&i_stats->rx_length_errors, output4, "rx_length_field_frame_errors");
    }

    return 0;
}

static int get_speed_mbps(const char *iface)
{
    char command[128];
    char output[4096];
    char *p;
    char *line;
    int speed = 0;

    memset(output, 0, sizeof(output));
    snprintf(command, sizeof(command), "ethtool %s", iface);

    if (run_command(command, output, sizeof(output)) != 0) {
        return 0;
    }

    p = strstr(output, "Speed:");
    if (!p) {
        return 0;
    }

    line = p + strlen("Speed:");
    while (*line == ' ' || *line == '\t') {
        line++;
    }

    speed = atoi(line);
    return speed;
}

static int read_host_temp(void)
{
    const char *paths[] = {
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
        "/sys/class/thermal/thermal_zone0/temp",
    };
    size_t i;

    for (i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        FILE *fp;
        int value = 0;

        fp = fopen(paths[i], "r");
        if (!fp) {
            continue;
        }

        if (fscanf(fp, "%d", &value) == 1) {
            fclose(fp);
            if (value > 200) {
                return value / 1000;
            }
            return value;
        }

        fclose(fp);
    }

    return 0;
}

static void update_anomaly_metrics(float mse, float threshold, double timestamp, cable_feature_state *feature_stats,
                                   cable_anomaly_history_metrics *anomaly_metrics) {
    int anomaly_level = NORMAL;

    if (mse <= 0.9 * threshold) {
        anomaly_level = NORMAL;
    } else if(mse > 0.9 * threshold &&
              mse <= threshold) {
        anomaly_level = SUSPICIOUS;
    } else {
        anomaly_level = ANOMALOUS;
    }

    /* push ot ANOMALOUS window*/
    if(anomaly_level == ANOMALOUS) {
        rolling_push(&feature_stats->anomalous_1h, timestamp, (int)1.0f, SPAN_1H);
    } else {
        rolling_push(&feature_stats->anomalous_1h, timestamp, (int)0.0f, SPAN_1H);
    }

    /* push ot SUSPICOUS window*/
    if(anomaly_level == SUSPICIOUS) {
        rolling_push(&feature_stats->suspicious_1h, timestamp, (int)1.0f, SPAN_1H);
    } else {
        rolling_push(&feature_stats->suspicious_1h, timestamp, (int)0.0f, SPAN_1H);
    }

    anomaly_metrics->anomalous_count_1h = rolling_sum(&feature_stats->anomalous_1h);
    anomaly_metrics->suspicious_count_1h = rolling_sum(&feature_stats->suspicious_1h);

}

int main() {
    int rc = 1;
    TfLiteModel* model = NULL;
    TfLiteInterpreterOptions* options = NULL;
    TfLiteDelegate* xnn_delegate = NULL;
    TfLiteInterpreter* interpreter = NULL;
    const int expected_inputs = NUM_INPUTS;
    const int expected_outputs = NUM_OUTPUTS;
    int in_count = 0;
    int out_count = 0;

    char stats_filename[PATH_MAX];
    char history_filename[PATH_MAX];

    cable_feature_state feature_state;
    struct interface_stats phy_stats;
    cable_features features;
    cable_history_metrics history_metrics;
    cable_anomaly_history_metrics anomaly_metrics;
    history_stats history_obj;
    float mse = 0.0f;

    memset(&history_obj, 0, sizeof(history_obj));
    memset(&feature_state, 0, sizeof(feature_state));
    memset(&phy_stats, 0, sizeof(phy_stats));
    memset(&features, 0, sizeof(features));
    memset(&history_metrics, 0, sizeof(history_metrics));
    memset(&anomaly_metrics, 0, sizeof(anomaly_metrics));
    memset(&config, 0, sizeof(config));
    memset(&stats_filename, 0, sizeof(stats_filename));
    memset(&history_filename, 0, sizeof(history_filename));

    if(read_config(CONFIG_PATH, &config)) {
        printf("Failed to load config from %s, trying from %s\n", CONFIG_PATH, CONFIG_PATH_ETC);

        if(read_config(CONFIG_PATH_ETC, &config)) {
            printf("Failed to load config from %s, exiting...\n", CONFIG_PATH_ETC);
            return 1;
        }
    }

    printf("======================Loaded config===============================\n");
    printf("-Model name: %s\n", config.model_name);
    printf("-Path: %s\n", config.model_path);
    printf("-Vocabulary path: %s\n", config.vocab_path);
    printf("-Enabled acceleration: %s\n", config.enabled_acceleration ? "true" : "false");
    printf("-xnpack number of threads: %d\n", config.xnnpack_num_threads);
    printf("-fallback number of threads: %d\n", config.fallback_num_threads);
    printf("-threshold: %f\n", config.threshold);
    printf("==================================================================\n");

    model = TfLiteModelCreateFromFile(config.model_path);
    if (!model) {
        printf("model load failed\n");
        goto cleanup;
    }

    options = TfLiteInterpreterOptionsCreate();
    if (!options) {
        printf("interpreter options create failed\n");
        goto cleanup;
    }

    TfLiteXNNPackDelegateOptions xnn_options = TfLiteXNNPackDelegateOptionsDefault();
    xnn_options.num_threads = config.xnnpack_num_threads; // adjust threads to your CPU

    if(config.enabled_acceleration) {
        xnn_delegate = TfLiteXNNPackDelegateCreate(&xnn_options);
        if (!xnn_delegate) {
            printf("xnn delegate create failed\n");
            goto cleanup;
        }

        TfLiteInterpreterOptionsAddDelegate(options, xnn_delegate);
        TfLiteInterpreterOptionsSetNumThreads(options, config.fallback_num_threads);
    }

    interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) {
        printf("interpreter create failed\n");
        goto cleanup;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        printf("allocate failed\n");
        goto cleanup;
    }

    // Validate model IO counts.
    in_count = TfLiteInterpreterGetInputTensorCount(interpreter);
    out_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
    if (in_count < expected_inputs || out_count < expected_outputs) {
        fprintf(stderr, "bad IO count: in=%d(out of %d), out=%d(out of %d)\n",
                in_count, expected_inputs, out_count, expected_outputs);
        goto cleanup;
    }

    model_inputs_outputs(interpreter);

    // sanity check inputs
    if (TfLiteTensorType(TfLiteInterpreterGetInputTensor(interpreter, 0)) != kTfLiteFloat32||
        TfLiteTensorByteSize(TfLiteInterpreterGetInputTensor(interpreter, 0)) != sizeof(float)*NUM_FEATURES) {
        fprintf(stderr, "input tensor type/size mismatch\n");
        goto cleanup;
    }
    // sanity check outputs
    if (TfLiteTensorType(TfLiteInterpreterGetOutputTensor(interpreter, 0)) != kTfLiteFloat32 ||
        TfLiteTensorByteSize(TfLiteInterpreterGetOutputTensor(interpreter, 0)) != sizeof(float)*NUM_FEATURES) {
        fprintf(stderr, "output tensor type/size mismatch\n");
        goto cleanup;
    }

    // ---- Inspect inputs (order matters) ----
    printf("Inputs: \n");
    for (int i = 0; i < in_count; ++i) {
        const TfLiteTensor* ti = TfLiteInterpreterGetInputTensor(interpreter, i);
        printf("input[%d] name=%s type=%d bytes=%zu\n", i, TfLiteTensorName(ti), TfLiteTensorType(ti), TfLiteTensorByteSize(ti));
    }

    mkdir(STATS_PATH, 0755);
    mkdir(HISTORY_PATH, 0755);

    sprintf(stats_filename, "%s/%s", STATS_PATH, STATS_FILENAME);
    sprintf(history_filename, "%s/%s", HISTORY_PATH, HISTORY_FILENAME);

    while (1) {
        raw_sample sample;
        int res;
        int anomaly_level;
        struct interface_stats port_stats;
        stats stats_obj;
        sample_history_record rec;
        float mse_per_feature[NUM_FEATURES];

        memset(&sample, 0, sizeof(sample));
        memset(&stats_obj, 0, sizeof(stats_obj));
        memset(&rec, 0, sizeof(rec));
        memset(&phy_stats, 0, sizeof(phy_stats));
        memset(&port_stats, 0, sizeof(port_stats));
        memset(&mse_per_feature, 0, sizeof(mse_per_feature));

        sample.ts_sec = (double)time(NULL);
        sample.carrier = read_int_sys(INTERFACE, "carrier");
        sample.speed_mbps = get_speed_mbps(INTERFACE);
        sample.temp_c = (float)read_host_temp();

        sample.rx_packets = read_u64_stat(INTERFACE, "rx_packets");
        sample.tx_packets = read_u64_stat(INTERFACE, "tx_packets");
        sample.rx_bytes = read_u64_stat(INTERFACE, "rx_bytes");
        sample.tx_bytes = read_u64_stat(INTERFACE, "tx_bytes");
        sample.rx_errors = read_u64_stat(INTERFACE, "rx_errors");
        sample.rx_crc_errors = read_u64_stat(INTERFACE, "rx_crc_errors");
        sample.tx_dropped = read_u64_stat(INTERFACE, "tx_dropped");

        if (ethtool_port_stats(INTERFACE, &port_stats) != 0) {
            memset(&port_stats, 0, sizeof(port_stats));
        }
        if (read_u64_stat_or_default(INTERFACE, "rx_frame_errors", &sample.rx_frame_errors) != 0) {
            sample.rx_frame_errors = port_stats.rx_frame_errors;
        }

        if (read_u64_stat_or_default(INTERFACE, "rx_length_errors", &sample.rx_length_errors) != 0) {
            sample.rx_length_errors = port_stats.rx_length_errors;
        }

        if (ethtool_phy_stats(INTERFACE, &phy_stats) != 0) {
            fprintf(stderr, "failed to read PHY stats\n");
            sleep(INTERVAL);
            continue;
        }
        sample.phy_receive_errors = phy_stats.phy_receive_errors;
        sample.phy_serdes_ber_errors = phy_stats.phy_serdes_ber_errors;

        res = invoke_model(interpreter, &feature_state, &sample, &features, &history_metrics, &mse, mse_per_feature);
        if (res == 1) {
            sleep(INTERVAL);
            continue;
        }
        if (res != 0) {
            fprintf(stderr, "invoke_model failed\n");
            sleep(INTERVAL);
            continue;
        }

        printf("reconstruction_error=%f\n", mse);

        if (mse <= 0.9 * config.threshold) {
            printf("result is %s [%f<%f] \n", states_string[NORMAL], mse, config.threshold);
        } else if(mse > 0.9 * config.threshold &&
                  mse<= config.threshold) {
            printf("result is %s [%f < %f < %f]\n", states_string[SUSPICIOUS], 0.9*config.threshold, mse, config.threshold);
        } else {
            printf("result is %s [%f<%f]\n", states_string[ANOMALOUS], config.threshold, mse);
        }

        update_anomaly_metrics(mse, config.threshold, sample.ts_sec, &feature_state, &anomaly_metrics);
        anomaly_level = classify_anomaly_level(mse, config.threshold);
        update_stats(&stats_obj, mse, anomaly_level);
        build_sample_history_record(&rec,
                                    sample.ts_sec,
                                    mse,
                                    anomaly_level,
                                    &features,
                                    &history_metrics,
                                    &anomaly_metrics,
                                    mse_per_feature,
                                    feature_names);
        history_push_record(&history_obj, &rec);
        if (save_stats_json(stats_filename, &stats_obj) != 0) {
            fprintf(stderr, "failed to save stats json\n");
        }
        if (save_history_json(history_filename, &history_obj) != 0) {
            fprintf(stderr, "failed to save history json\n");
        }

        sleep(INTERVAL);
    }
    rc = 0;

cleanup:
    if (interpreter) TfLiteInterpreterDelete(interpreter);
    if (options) TfLiteInterpreterOptionsDelete(options);
    if (xnn_delegate) TfLiteXNNPackDelegateDelete(xnn_delegate);
    if (model) TfLiteModelDelete(model);

    return rc;
}
