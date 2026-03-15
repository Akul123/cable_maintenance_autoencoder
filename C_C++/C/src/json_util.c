#include <json-c/json.h>
#include <string.h>
#include "json_util.h"

extern const char *states_string[];

int classify_anomaly_level(float mse, float threshold) {
    if (mse <= 0.9f * threshold) {
        return NORMAL;
    }
    if (mse <= threshold) {
        return SUSPICIOUS;
    }
    return ANOMALOUS;
}

void update_stats(stats *s, float mse, int anomaly_level) {
    int was_anomalous;

    if (!s) return;

    was_anomalous = (s->last_anomaly_level == ANOMALOUS);

    s->total_samples++;
    s->last_mse = mse;

    if (s->total_samples == 1 || mse > s->max_mse_seen) {
        s->max_mse_seen = mse;
    }

    if (anomaly_level == NORMAL) {
        s->total_normal_count++;
    } else if (anomaly_level == SUSPICIOUS) {
        s->total_suspicious_count++;
    } else if (anomaly_level == ANOMALOUS) {
        s->total_anomalous_count++;
        if (!was_anomalous) {
            s->total_anomaly_events++;
        }
    }

    s->last_anomaly_level = anomaly_level;
}

void history_push_record(history_stats *h, const sample_history_record *rec) {
    size_t idx;

    if (!h || !rec) return;

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
    rec->anomalous_count_1h = anomaly_metrics->anomalous_count_1h;
    rec->suspicious_count_1h = anomaly_metrics->suspicious_count_1h;

    rec->speed_change_count_10m = features->speed_change_count_10m;
    rec->flaps_10m = features->flaps_10m;
    rec->temp_slope_10m = features->temp_slope_10m;

    rec->speed_change_count_1h = history_metrics->speed_change_count_1h;
    rec->flaps_1h = history_metrics->flaps_1h;
    rec->temp_slope_1h = history_metrics->temp_slope_1h;

    fill_top3_features(rec, feature_errors, feature_names, NUM_FEATURES);
}

int save_stats_json(const char *path, const stats *s) {
    struct json_object *root;
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

    fprintf(stderr, "save_stats_json: writing to %s\n", path);

    rc = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    if (rc != 0) {
        perror("json_object_to_file_ext");
        fprintf(stderr, "save_stats_json: failed to write %s\n", path);
    }

    json_object_put(root);
    return rc;
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
        if (root) json_object_put(root);
        if (records) json_object_put(records);
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
        json_object_object_add(item, "temp_slope_10m", json_object_new_double(rec->temp_slope_10m));
        json_object_object_add(item, "speed_change_count_1h", json_object_new_double(rec->speed_change_count_1h));
        json_object_object_add(item, "flaps_1h", json_object_new_double(rec->flaps_1h));
        json_object_object_add(item, "temp_slope_1h", json_object_new_double(rec->temp_slope_1h));

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

    fprintf(stderr, "save_history_json: writing to %s\n", path);

    rc = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    if (rc != 0) {
        perror("json_object_to_file_ext");
        fprintf(stderr, "save_history_json: failed to write %s\n", path);
    }

    json_object_put(root);
    return rc;
}