#include <json-c/json.h>
#include <string.h>
#include "json_util.h"
#include "time.h"
#include "rolling_window.h"

extern const char *states_string[];

const char *radar_labels[NUM_FEATURES] = {
    "Rx Frame Errors / 1M Packets",
    "Rx Length Errors / 1M Packets",
    "Speed Changes / 10min",
    "Speed Downgrade",
    "RX Errors",
    "PHY RX Errors",
    "SerDes BER Errors",
    "Rx CRC Errors / 1M Packets",
    "RX Error Rate",
    "Local Receiver NOK",
    "Remote Receiver NOK",
    //"Avg FCS / 1M",
    "Utilization",
    "Link Flaps / 10min",
    "Temp Trend / 10min"
};

int classify_anomaly_level(float mse, float threshold) {
    if (mse <= 0.9f * threshold) {
        return NORMAL;
    }
    if (mse <= threshold) {
        return SUSPICIOUS;
    }
    return ANOMALOUS;
}

// // Fast short-term memory:
// // F_t = alpha_f * e_t + (1 - alpha_f) * F_(t-1)

// // Slow long-term memory:
// // S_t = alpha_s * e_t + (1 - alpha_s) * S_(t-1)

// // Constraint:
// // 0 < alpha_s < alpha_f < 1

// // Persistent degradation pressure from slow trend:
// // x_t = max(0, S_t - beta * T)

// // Long-term degradation score:
// // D_t = lambda * D_(t-1) + x_t

// // where:
// // lambda = 1.0           // no forgetting
// // 0.999 <= lambda < 1.0  // very slow forgetting

// // Alternative pressure using fast-above-slow trend:
// // x_t = max(0, F_t - S_t)

// // Then:
// // D_t = lambda * D_(t-1) + x_t

// // Combined version:
// // F_t = alpha_f * e_t + (1 - alpha_f) * F_(t-1)
// // S_t = alpha_s * e_t + (1 - alpha_s) * S_(t-1)
// // x_t = max(0, S_t - beta * T)
// // y_t = max(0, F_t - S_t)
// // D_t = lambda * D_(t-1) + w1 * x_t + w2 * y_t
// static void update_degradation_metrics(stats *s, float mse, float threshold) {
//     float x_t, y_t;

//     if (!s) {
//         return;
//     }

//     if (!s->ewma_initialized) {
//         s->ewma_fast = mse;
//         s->ewma_slow = mse;
//         s->degradation_score = 0.0f;
//         s->ewma_initialized = true;
//         return;
//     }

//     s->ewma_fast = ALPHA_FAST * mse + (1.0f - ALPHA_FAST) * s->ewma_fast;
//     s->ewma_slow = ALPHA_SLOW * mse + (1.0f - ALPHA_SLOW) * s->ewma_slow;

//     x_t = fmaxf(0.0f, s->ewma_slow - BETA * threshold); // long term memory component
//     y_t = fmaxf(0.0f, s->ewma_fast - s->ewma_slow);     // short term memory component

//     s->degradation_score = LAMBDA * s->degradation_score + W1 * x_t + W2 * y_t;
// }

// Degradation score with fast reaction and long memory.
//
// We first compress mse with log1p() so rare extreme spikes do not dominate
// the score. Then we track two EWMA signals:
//
// F_t: fast EWMA, reacts within minutes to fresh anomalies
// S_t: slow EWMA, represents longer-term degradation trend
//
// From those we derive two non-negative pressures:
//
// slow_pressure = max(0, S_t - beta * T)
//   Measures sustained degradation above the threshold baseline.
//
// fast_pressure = max(0, F_t - S_t)
//   Measures new short-term worsening relative to the long-term trend.
//
// The total pressure is a weighted blend of long-term and short-term effects:
//
// instant_pressure = w_long * slow_pressure + w_short * fast_pressure
//
// Finally, degradation_score is updated as a leaky accumulator:
//
// D_t = memory * D_(t-1) + gain * instant_pressure
//
// This makes the score rise quickly when cable quality worsens, remember
// degradation for hours after the event, and decay gradually once pressure
// disappears instead of staying elevated forever.
static void update_degradation_metrics(stats *s, float mse, float threshold) {
    float mse_log;
    float threshold_log;
    float slow_pressure;
    float fast_pressure;
    float instant_pressure;

    if (!s) {
        return;
    }

    mse_log = log1pf(fmaxf(0.0f, mse));
    threshold_log = log1pf(fmaxf(0.0f, threshold));

    if (!s->ewma_initialized) {
        s->ewma_fast = mse_log;
        s->ewma_slow = mse_log;
        s->degradation_score = 0.0f;
        s->ewma_initialized = 1;
        return;
    }

    s->ewma_fast = DEG_ALPHA_FAST * mse_log + (1.0f - DEG_ALPHA_FAST) * s->ewma_fast;
    s->ewma_slow = DEG_ALPHA_SLOW * mse_log + (1.0f - DEG_ALPHA_SLOW) * s->ewma_slow;

    slow_pressure = fmaxf(0.0f, s->ewma_slow - DEG_BETA * threshold_log);
    fast_pressure = fmaxf(0.0f, s->ewma_fast - s->ewma_slow);
    instant_pressure = DEG_W_LONG * slow_pressure + DEG_W_SHORT * fast_pressure;

    s->degradation_score = DEG_MEMORY * s->degradation_score + DEG_GAIN * instant_pressure;
}

void update_stats(stats *s, float mse, int anomaly_level, const char* reason, int max_mse_index, float threshold) {
    int was_anomalous;
    double timestamp = (double)time(NULL);

    if (!s) return;

    was_anomalous = (s->last_anomaly_level == ANOMALOUS);

    s->total_samples++;
    s->last_mse = mse;
    s->threshold = threshold;

    if (s->total_samples == 1 || mse > s->max_mse_seen) {
        s->max_mse_seen = mse;
    }

    /* anomaly_event records */
    if (anomaly_level == NORMAL) {
        s->total_normal_count++;
        anomaly_event_push(&s->samples, timestamp, mse, NORMAL, "normal sample");
    } else if (anomaly_level == SUSPICIOUS) {
        s->total_suspicious_count++;
        anomaly_event_push(&s->samples, timestamp, mse, SUSPICIOUS, reason);
    } else if (anomaly_level == ANOMALOUS) {
        s->total_anomalous_count++;
        anomaly_event_push(&s->samples, timestamp, mse, ANOMALOUS, reason);
        if (!was_anomalous) {
            s->total_anomaly_events++;
        }
    }

    /* history score records */
    // float sample_term = 0.3f * log1pf(fmaxf(mse, 0.0f));
    float history_term =
        0.20f * log1pf((float)s->total_suspicious_count) +
        0.60f * log1pf((float)s->total_anomalous_count) +
        0.80f * log1pf((float)s->total_anomaly_events);

    // float val = sample_term + history_term;
    float val = history_term;

    score_push(&s->history_score, timestamp, val);

    // in case of anomaly sample increment count
    // for feature with biggest mse
    if(anomaly_level==ANOMALOUS)
        s->feature_counts[max_mse_index]++;

    s->last_anomaly_level = anomaly_level;

    // degradations score
    update_degradation_metrics(s, mse, threshold);
    score_push(&s->degradation_score_window, timestamp, s->degradation_score);
}

void history_push_record(history_stats *h, const sample_history_record *rec) {
    size_t idx;

    if (!h || !rec)
        return;

    if (h->count == WINDOW_CAP) {
        h->start = (h->start + 1) % WINDOW_CAP;
        h->count--;
    }

    idx = (h->start + h->count) % WINDOW_CAP;
    h->records[idx] = *rec;
    h->count++;
}

void fill_top3_features(sample_history_record *rec,
                        const float *feature_errors,
                        const char *const *feature_names,
                        size_t feature_count) {
    size_t i, j;

    if (!rec || !feature_errors || !feature_names) return;

    for (i = 0; i < 3; ++i) {
        rec->top3[i].name[0] = '\0';
        rec->top3[i].error = -1.0f;
    }

    for (i = 0; i < feature_count; ++i) {
        float err = feature_errors[i];

        for (j = 0; j < 3; ++j) {
            if (err > rec->top3[j].error) {
                size_t k;
                for (k = 2; k > j; --k) {
                    rec->top3[k] = rec->top3[k - 1];
                }

                strncpy(rec->top3[j].name, feature_names[i], sizeof(rec->top3[j].name) - 1);
                rec->top3[j].name[sizeof(rec->top3[j].name) - 1] = '\0';
                rec->top3[j].error = err;
                break;
            }
        }
    }
}

void build_sample_history_record(sample_history_record *rec,
                                 double ts_sec,
                                 float mse,
                                 int anomaly_level,
                                 const char *reason,
                                 const cable_features *features,
                                 const cable_history_metrics *history_metrics,
                                 const cable_anomaly_history_metrics *anomaly_metrics,
                                 const float *feature_errors,
                                 const char *const *feature_names) {
    if (!rec || !features || !history_metrics || !feature_errors || !feature_names)
        return;

    memset(rec, 0, sizeof(*rec));

    rec->ts_sec = ts_sec;
    rec->mse = mse;
    rec->anomaly_level = anomaly_level;
    /* reason */
    if (rec->anomaly_level == NORMAL)
        strcpy(rec->reason, "normal");
    else
        strcpy(rec->reason, reason);

    rec->anomalous_count_1h = anomaly_metrics->anomalous_count_1h;
    rec->suspicious_count_1h = anomaly_metrics->suspicious_count_1h;

    rec->speed_change_count_10m = features->speed_change_count_10m;
    rec->flaps_10m = features->flaps_10m;
    rec->temp_delta_10m = features->temp_delta_10m;

    rec->speed_change_count_1h = history_metrics->speed_change_count_1h;
    rec->flaps_1h = history_metrics->flaps_1h;
    rec->temp_delta_1h = history_metrics->temp_delta_1h;

    fill_top3_features(rec, feature_errors, feature_names, NUM_FEATURES);
}

static void feature_to_json(struct json_object *feature_counts_json, const char *name, uint16_t val) {
    json_object *item = json_object_new_object();
    json_object_object_add(item, "name", json_object_new_string(name));
    json_object_object_add(item, "val", json_object_new_double(val));
    json_object_array_add(feature_counts_json, item);
}

int save_stats_json(const char *path, const stats *s) {
    struct json_object *root;
    struct json_object *stats_samples_records, *stats_history_score_records, *feature_counts, *degradation_score_records;
    int rc;

    if (!path || !s) return -1;

    root = json_object_new_object();
    if (!root) {
        return -1;
    }

    json_object_object_add(root, "total_samples", json_object_new_int64((int64_t)s->total_samples));
    json_object_object_add(root, "total_normal_count", json_object_new_int64((int64_t)s->total_normal_count));
    json_object_object_add(root, "total_suspicious_count", json_object_new_int64((int64_t)s->total_suspicious_count));
    json_object_object_add(root, "total_anomalous_count", json_object_new_int64((int64_t)s->total_anomalous_count));
    json_object_object_add(root, "total_anomaly_events", json_object_new_int64((int64_t)s->total_anomaly_events));
    json_object_object_add(root, "max_mse_seen", json_object_new_double((double)s->max_mse_seen));
    json_object_object_add(root, "last_mse", json_object_new_double((double)s->last_mse));
    json_object_object_add(root, "last_anomaly_level", json_object_new_string(states_string[s->last_anomaly_level]));
    json_object_object_add(root, "threshold", json_object_new_double((double)s->threshold));
    json_object_object_add(root, "ewma_initialized", json_object_new_int(s->ewma_initialized));
    json_object_object_add(root, "degradation_score", json_object_new_double(s->degradation_score));
    json_object_object_add(root, "ewma_slow", json_object_new_double(s->ewma_slow));
    json_object_object_add(root, "ewma_fast", json_object_new_double(s->ewma_fast));
    //json_object_object_add(root, "current_ewma_error", json_object_new_double((double)s->current_ewma_error));

    /* samples records */
    stats_samples_records = json_object_new_array();
    for (size_t i = 0; i < s->samples.count; ++i) {
        size_t idx = (s->samples.start + i) % EVENT_WINDOW_CAP;
        struct json_object *item = json_object_new_object();
        json_object_object_add(item, "ts", json_object_new_double(s->samples.records[idx].record.ts));
        json_object_object_add(item, "val", json_object_new_double(s->samples.records[idx].record.val));
        json_object_object_add(item, "reason", json_object_new_string(s->samples.records[idx].reason));
        json_object_object_add(item, "classifier", json_object_new_int(s->samples.records[idx].classifier));
        json_object_array_add(stats_samples_records, item);
    }
    json_object_object_add(root, "samples_records", stats_samples_records);

    /* degradation_score_window records */
    degradation_score_records = json_object_new_array();
    for (size_t i = 0; i < s->degradation_score_window.count; ++i) {
        size_t idx = (s->degradation_score_window.start + i) % EVENT_WINDOW_CAP;
        struct json_object *item = json_object_new_object();
        json_object_object_add(item, "ts", json_object_new_double(s->degradation_score_window.records[idx].ts));
        json_object_object_add(item, "val", json_object_new_double(s->degradation_score_window.records[idx].val));
        json_object_array_add(degradation_score_records, item);
    }
    json_object_object_add(root, "degradations_score_records", degradation_score_records);

    /* history score records */
    stats_history_score_records = json_object_new_array();
    for (size_t i = 0; i < s->history_score.count; ++i) {
        size_t idx = (s->history_score.start + i) % EVENT_WINDOW_CAP;
        struct json_object *item = json_object_new_object();
        json_object_object_add(item, "ts", json_object_new_double(s->history_score.records[idx].ts));
        json_object_object_add(item, "val", json_object_new_double(s->history_score.records[idx].val));
        json_object_array_add(stats_history_score_records, item);
    }
    json_object_object_add(root, "history_score", stats_history_score_records);

    /* feature counts */
    // feature_counts = json_object_new_array();
    // for (size_t i = 0; i < NUM_FEATURES; ++i) {
    //     struct json_object *item = json_object_new_object();
    //     json_object_object_add(item, "name", json_object_new_string(feature_display_names[i]));
    //     json_object_object_add(item, "val", json_object_new_int(s->feature_counts[i]));
    //     json_object_array_add(feature_counts, item);
    // }
    // json_object_object_add(root, "feature_counts", feature_counts);
// const char *const feature_display_names[NUM_FEATURES] = {
//     "Frame Errors",
//     "Length Errors",
//     "Speed Changes (10m)",
//     "Speed Downgrade",
//     "RX Errors",
//     "PHY RX Errors",
//     "SerDes BER Errors",
//     "FCS / 1M Packets",
//     "RX Error Rate",
//     "Local Receiver NOK",
//     "Remote Receiver NOK",
//     "Avg FCS / 1M",
//     "Max FCS / 1M",
//     "Utilization",
//     "Link Flaps (10m)",
//     "Temp Trend (10m)"
// };
// const char* const radar_labels = {
//     "Frame Errors",
//     "Length Errors",
//     "RX Errors",
//     "PHY RX Errors",
//     "SerDes BER Errors",
//     "FCS / 1M Packets",
//     "RX Error Rate",
//     "Local Receiver NOK",
//     "Remote Receiver NOK",
//     "Utilization"
// };
    // map between feature_display_names & radar_labels
    static const int radar_feature_idx[NUM_FEATURES] = {
        0,   /* Frame Errors */
        1,   /* Length Errors */
        2,   /* Speed Changes (10m) */
        3,   /* Speed Downgrade */
        4,   /* RX Errors */
        5,   /* PHY RX Errors */
        6,   /* SerDes BER Errors */
        7,   /* FCS / 1M Packets */
        8,   /* RX Error Rate */
        9,   /* Local Receiver NOK */
        10,  /* Remote Receiver NOK */
        //11,  /* mean_fcs_per_million*/
        11,  /* Utilization */
        12,  /* flaps_10min */
        13   /* temp_10min */
    };

    feature_counts = json_object_new_array();
    for (size_t i = 0; i < sizeof(s->feature_counts)/sizeof(uint64_t); ++i) {
        // feature_to_json(feature_counts, radar_labels[i], s->feature_counts[radar_feature_idx[i]]);
        feature_to_json(feature_counts, radar_labels[i], s->feature_counts[i]);
    }
    json_object_object_add(root, "feature_counts", feature_counts);

    rc = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    if (rc != 0) {
        perror("json_object_to_file_ext");
        fprintf(stderr, "save_stats_json: failed to write %s\n", path);
    }

    json_object_put(root);
    return rc;
}

static int parse_anomaly_level_string(const char *level) {
    if (!level) return NORMAL;
    if (strcmp(level, "normal") == 0) return NORMAL;
    if (strcmp(level, "suspicious") == 0) return SUSPICIOUS;
    if (strcmp(level, "anomalous") == 0) return ANOMALOUS;
    return NORMAL;
}

int load_stats_json(const char *path, stats *out) {
    struct json_object *root = NULL;
    struct json_object *obj = NULL;
    struct json_object *arr = NULL;
    size_t i;

    if (!path || !out) {
        return -1;
    }

    root = json_object_from_file(path);
    if (!root) {
        return -1;
    }

    if (json_object_object_get_ex(root, "ewma_initialized", &obj)) {
        out->ewma_initialized = json_object_get_boolean(obj);
    }

    // if (json_object_object_get_ex(root, "current_ewma_error", &obj)) {
    //     out->current_ewma_error = (float)json_object_get_double(obj);
    // }

    if (json_object_object_get_ex(root, "degradation_score", &obj)) {
        out->degradation_score = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "ewma_fast", &obj)) {
        out->ewma_fast = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "ewma_slow", &obj)) {
        out->ewma_slow = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "total_samples", &obj)) {
        out->total_samples = (uint64_t)json_object_get_int64(obj);
    }

    if (json_object_object_get_ex(root, "total_normal_count", &obj)) {
        out->total_normal_count = (uint64_t)json_object_get_int64(obj);
    }

    if (json_object_object_get_ex(root, "total_suspicious_count", &obj)) {
        out->total_suspicious_count = (uint64_t)json_object_get_int64(obj);
    }

    if (json_object_object_get_ex(root, "total_anomalous_count", &obj)) {
        out->total_anomalous_count = (uint64_t)json_object_get_int64(obj);
    }

    if (json_object_object_get_ex(root, "total_anomaly_events", &obj)) {
        out->total_anomaly_events = (uint64_t)json_object_get_int64(obj);
    }

    if (json_object_object_get_ex(root, "max_mse_seen", &obj)) {
        out->max_mse_seen = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "last_mse", &obj)) {
        out->last_mse = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "threshold", &obj)) {
        out->threshold = (float)json_object_get_double(obj);
    }

    if (json_object_object_get_ex(root, "last_anomaly_level", &obj) &&
        json_object_is_type(obj, json_type_string)) {
        out->last_anomaly_level = parse_anomaly_level_string(json_object_get_string(obj));
    }

    if (json_object_object_get_ex(root, "samples_records", &arr) &&
        json_object_is_type(arr, json_type_array)) {
        size_t n = json_object_array_length(arr);
        out->samples.start = 0;
        out->samples.count = 0;

        for (i = 0; i < n && i < EVENT_WINDOW_CAP; ++i) {
            struct json_object *item = json_object_array_get_idx(arr, i);
            struct json_object *ts = NULL, *val = NULL, *reason = NULL, *classifier = NULL;
            size_t idx = out->samples.count;

            if (!item) continue;

            json_object_object_get_ex(item, "ts", &ts);
            json_object_object_get_ex(item, "val", &val);
            json_object_object_get_ex(item, "reason", &reason);
            json_object_object_get_ex(item, "classifier", &classifier);

            out->samples.records[idx].record.ts = ts ? json_object_get_double(ts) : 0.0;
            out->samples.records[idx].record.val = val ? (float)json_object_get_double(val) : 0.0f;
            out->samples.records[idx].classifier = classifier ? json_object_get_int(classifier) : NORMAL;

            if (reason && json_object_is_type(reason, json_type_string)) {
                strncpy(out->samples.records[idx].reason,
                        json_object_get_string(reason),
                        REASON_LEN - 1);
                out->samples.records[idx].reason[REASON_LEN - 1] = '\0';
            } else {
                out->samples.records[idx].reason[0] = '\0';
            }

            out->samples.count++;
        }
    }

    if (json_object_object_get_ex(root, "history_score", &arr) &&
        json_object_is_type(arr, json_type_array)) {
        size_t n = json_object_array_length(arr);
        out->history_score.start = 0;
        out->history_score.count = 0;

        for (i = 0; i < n && i < EVENT_WINDOW_CAP; ++i) {
            struct json_object *item = json_object_array_get_idx(arr, i);
            struct json_object *ts = NULL, *val = NULL;
            size_t idx = out->history_score.count;

            if (!item) continue;

            json_object_object_get_ex(item, "ts", &ts);
            json_object_object_get_ex(item, "val", &val);

            out->history_score.records[idx].ts = ts ? json_object_get_double(ts) : 0.0;
            out->history_score.records[idx].val = val ? (float)json_object_get_double(val) : 0.0f;
            out->history_score.count++;
        }
    }

    if (json_object_object_get_ex(root, "feature_counts", &arr) &&
        json_object_is_type(arr, json_type_array)) {
        size_t n = json_object_array_length(arr);

        for (i = 0; i < n && i < NUM_FEATURES; ++i) {
            struct json_object *item = json_object_array_get_idx(arr, i);
            struct json_object *val = NULL;

            if (!item) continue;
            if (json_object_object_get_ex(item, "val", &val)) {
                out->feature_counts[i] = (uint64_t)json_object_get_int64(val);
            }
        }
    }

    // if (json_object_object_get_ex(root, "ewma_error_records", &arr) &&
    //     json_object_is_type(arr, json_type_array)) {
    //     size_t n = json_object_array_length(arr);

    //     for (i = 0; i < n; ++i) {
    //         struct json_object *item = json_object_array_get_idx(arr, i);
    //         struct json_object *val = NULL, *ts = NULL;
    //         size_t idx = out->ewma_records.count;

    //         if (!item) continue;

    //         json_object_object_get_ex(item, "ts", &ts);
    //         json_object_object_get_ex(item, "val", &val);

    //         if (json_object_object_get_ex(item, "val", &val)) {
    //             out->ewma_records.records[i].val = (double)json_object_get_double(val);
    //             out->ewma_records.records[i].ts = ts ? json_object_get_double(ts) : 0.0;
    //         }
    //     }
    // }

    if (json_object_object_get_ex(root, "degradations_score_records", &arr) &&
        json_object_is_type(arr, json_type_array)) {
        size_t n = json_object_array_length(arr);
        out->degradation_score_window.start = 0;
        out->degradation_score_window.count = 0;

        for (i = 0; i < n && i < EVENT_WINDOW_CAP; ++i) {
            struct json_object *item = json_object_array_get_idx(arr, i);
            struct json_object *val = NULL, *ts = NULL;
            size_t idx = out->degradation_score_window.count;

            if (!item) continue;

            json_object_object_get_ex(item, "ts", &ts);
            json_object_object_get_ex(item, "val", &val);

            if (json_object_object_get_ex(item, "val", &val)) {
                out->degradation_score_window.records[idx].val = (double)json_object_get_double(val);
                out->degradation_score_window.records[idx].ts = ts ? json_object_get_double(ts) : 0.0;
            }
            out->degradation_score_window.count++;
        }
    }

    json_object_put(root);

    return 0;
}

int save_history_json(const char *path, const history_stats *h) {
    struct json_object *root;
    struct json_object *records;
    size_t i;
    int rc;

    if (!path || !h) return -1;

    root = json_object_new_object();
    records = json_object_new_array();
    if (!root || !records) {
        json_object_put(root);
        json_object_put(records);
        return -1;
    }

    for (i = 0; i < h->count; ++i) {
        size_t idx = (h->start + i) % WINDOW_CAP;
        const sample_history_record *rec = &h->records[idx];
        struct json_object *item = json_object_new_object();
        struct json_object *top3 = json_object_new_array();
        size_t j;

        if (!item || !top3) {
            if (item) json_object_put(item);
            if (top3) json_object_put(top3);
            json_object_put(records);
            json_object_put(root);
            return -1;
        }

        json_object_object_add(item, "ts_sec", json_object_new_double(rec->ts_sec));
        json_object_object_add(item, "mse", json_object_new_double(rec->mse));
        json_object_object_add(item, "anomaly_level", json_object_new_string(states_string[rec->anomaly_level]));
        json_object_object_add(item, "speed_change_count_10m", json_object_new_double(rec->speed_change_count_10m));
        json_object_object_add(item, "flaps_10m", json_object_new_double(rec->flaps_10m));
        json_object_object_add(item, "temp_delta_10m", json_object_new_double(rec->temp_delta_10m));
        json_object_object_add(item, "speed_change_count_1h", json_object_new_double(rec->speed_change_count_1h));
        json_object_object_add(item, "flaps_1h", json_object_new_double(rec->flaps_1h));
        json_object_object_add(item, "temp_delta_1h", json_object_new_double(rec->temp_delta_1h));
        json_object_object_add(item, "anomalous_count_1h", json_object_new_double(rec->anomalous_count_1h));
        json_object_object_add(item, "suspicious_count_1h", json_object_new_double(rec->suspicious_count_1h));
        json_object_object_add(item, "reason", json_object_new_string(rec->reason));

        for (j = 0; j < 3; ++j) {
            struct json_object *top = json_object_new_object();
            if (!top) {
                json_object_put(item);
                json_object_put(top3);
                json_object_put(records);
                json_object_put(root);
                return -1;
            }

            json_object_object_add(top, "name", json_object_new_string(rec->top3[j].name));
            json_object_object_add(top, "error", json_object_new_double(rec->top3[j].error));
            json_object_array_add(top3, top);
        }

        json_object_object_add(item, "top3", top3);
        json_object_array_add(records, item);
    }

    json_object_object_add(root, "count", json_object_new_int((int)h->count));
    json_object_object_add(root, "records", records);

    rc = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    if (rc != 0) {
        perror("json_object_to_file_ext");
        fprintf(stderr, "save_history_json: failed to write %s\n", path);
    }

    json_object_put(root);
    return rc;
}