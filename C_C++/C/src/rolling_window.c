#include <math.h>
#include <string.h>

#include "rolling_window.h"

void rolling_push(rolling_window *w, double ts, float value, double span_sec) {
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

void rolling_push_fixed(rolling_window *w, double ts, float value) {
    size_t end;

    if (!w) {
        return;
    }

    /* If full, drop the oldest element */
    if (w->count == WINDOW_CAP) {
        w->start = (w->start + 1) % WINDOW_CAP;
        w->count--;
    }

    /* Append new element at logical end */
    end = (w->start + w->count) % WINDOW_CAP;
    w->ts[end] = ts;
    w->val[end] = value;
    w->count++;
}

float rolling_sum(const rolling_window *w) {
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx])) sum += w->val[idx];
    }
    return sum;
}

float rolling_first(const rolling_window *w) {
    size_t i;
    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx])) return w->val[idx];
    }
    return NAN;
}

float rolling_avg(const rolling_window *w) {
    float sum = 0.0f;
    int sum_count = 0;
    size_t i;

    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx])) {
            sum += w->val[idx];
            sum_count++;
        }
    }

    if(sum_count==0)
        return NAN;
    else
        return sum/sum_count;
}

float rolling_max(const rolling_window *w) {
    int found = 0;
    float max = 0.0f;
    size_t i;

    for (i = 0; i < w->count; ++i) {
        size_t idx = (w->start + i) % WINDOW_CAP;
        if (!isnan(w->val[idx]) && (!found || w->val[idx] > max)) {
            max = w->val[idx];
            found = 1;
        }
    }

    return found ? max : NAN;
}

void anomaly_event_push(anomaly_event_window *w,
                        double ts_sec,
                        float val,
                        const int classification,
                        const char *reason) {
    size_t idx;

    if (!w) return;

    if (w->count == EVENT_WINDOW_CAP) {
        w->start = (w->start + 1) % EVENT_WINDOW_CAP;
        w->count--;
    }

    idx = (w->start + w->count) % EVENT_WINDOW_CAP;
    w->records[idx].ts = ts_sec;
    w->records[idx].val = val;
    w->records[idx].classifier = classification;

    if (reason) {
        strncpy(w->records[idx].reason, reason, REASON_LEN - 1);
        w->records[idx].reason[REASON_LEN - 1] = '\0';
    } else {
        w->records[idx].reason[0] = '\0';
    }

    w->count++;
}