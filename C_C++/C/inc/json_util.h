#ifndef JSON_UTIL_H
#define JSON_UTIL_H

#include <stddef.h>
#include <stdint.h>

#include "../cable_autoencoder_xnnpack.h"


typedef struct top_feature_entry {
    char name[64];
    float error;
} top_feature_entry;

typedef struct sample_history_record {
    double ts_sec;
    float mse;
    int anomaly_level;
    int anomalous_count_1h;
    int suspicious_count_1h;

    float speed_change_count_10m;
    float flaps_10m;
    float temp_slope_10m;

    float speed_change_count_1h;
    float flaps_1h;
    float temp_slope_1h;

    top_feature_entry top3[3];
} sample_history_record;

typedef struct history_stats {
    sample_history_record records[256];
    size_t start;
    size_t count;
} history_stats;

typedef struct stats {
    uint64_t total_samples;
    uint64_t total_normal_count;
    uint64_t total_suspicious_count;
    uint64_t total_anomalous_count;
    uint64_t total_anomaly_events;
    float max_mse_seen;
    float last_mse;
    int last_anomaly_level;
} stats;

int classify_anomaly_level(float mse, float threshold);

void update_stats(stats *s, float mse, int anomaly_level);
void history_push_record(history_stats *h, const sample_history_record *rec);

void fill_top3_features(sample_history_record *rec,
                        const float *feature_errors,
                        const char *const *feature_names,
                        size_t feature_count);

void build_sample_history_record(sample_history_record *rec,
                                 double ts_sec,
                                 float mse,
                                 int anomaly_level,
                                 const cable_features *features,
                                 const cable_history_metrics *history_metrics,
                                 const cable_anomaly_history_metrics *anomaly_metrics,
                                 const float *feature_errors,
                                 const char *const *feature_names);

int load_stats_json(const char *path, stats *out);
int save_stats_json(const char *path, const stats *s);

int load_history_json(const char *path, history_stats *out);
int save_history_json(const char *path, const history_stats *h);

#endif
