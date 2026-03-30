#ifndef ROLLING_WINDOW_H
#define ROLLING_WINDOW_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define WINDOW_CAP          256     // number timestamps which is being tracked
#define EVENT_WINDOW_CAP    7200    // around 5 days in minutes
#define REASON_LEN          128

typedef struct rolling_window {
    double ts[WINDOW_CAP];  // timestamp array
    float val[WINDOW_CAP];  // value array
    size_t start;
    size_t count;
} rolling_window;

typedef struct history_event {
    double ts;
    float val;
} history_event;

typedef struct anomaly_event_record {
    //double ts;
    //float val;      // MSE value
    history_event event;
    int classifier; // NORMAL || SUSPICIOUS || ANOMALOUS
    char reason[REASON_LEN];
} anomaly_event_record;

typedef struct anomaly_event_window {
    anomaly_event_record records[EVENT_WINDOW_CAP];
    size_t start;
    size_t count;
} anomaly_event_window;

typedef struct history_score_window {
    history_event records[EVENT_WINDOW_CAP];
    size_t start;
    size_t count;
} history_score_window;

typedef struct top_feature_entry {
    char name[64];
    float error;
} top_feature_entry;

// anomaly window
void anomaly_event_push(anomaly_event_window *w, double ts_sec, float val, const int classification, const char *reason);

// history score window
void history_score_push(history_score_window *w, double ts_sec, float val);

// rolling window
void rolling_push(rolling_window *w, double ts, float value, double span_sec);
void rolling_push_fixed(rolling_window *w, double ts, float value);
float rolling_sum(const rolling_window *w);
float rolling_first(const rolling_window *w);
float rolling_avg(const rolling_window *w);
float rolling_max(const rolling_window *w);

#endif // ROLLING_WINDOW_H