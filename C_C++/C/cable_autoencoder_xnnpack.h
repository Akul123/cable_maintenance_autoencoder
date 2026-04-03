#ifndef CABLE_AUTOENCODER_XNNPACK_H
#define CABLE_AUTOENCODER_XNNPACK_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "rolling_window.h"

#define NUM_INPUTS      1
#define NUM_OUTPUTS     1
#define NUM_FEATURES    14
#define INTERVAL        60
#define BITS_PER_BYTE   8.0f
#define ALPHA           0.05f // EWMA smoothing factor, smaller for longer timeperiods

#define SPAN_10MIN      600.0f  // 10 min in seconds
#define SPAN_1H         3600.0f // 1 hour in seconds

#define STATS_PATH          "/var/lib/cable_autoencoder_xnnpack"
#define STATS_FILENAME      "stats.json"
#define HISTORY_PATH        "/var/lib/cable_autoencoder_xnnpack"
#define HISTORY_FILENAME    "history.json"

#if DEBUG
#define LOGF(...) printf(__VA_ARGS__)
#else
#define LOGF(...) do {} while (0)
#endif

#define CONFIG_PATH     "./config.json"
#define CONFIG_PATH_ETC "/etc/cable_autoencoder/model_config.json"
#define INTERFACE       "eth0"

#define NORMAL      0
#define SUSPICIOUS  1
#define ANOMALOUS   2

struct interface_stats {
    int32_t peer_ifindex;

    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t rx_bytes;
    uint64_t tx_bytes;

    uint64_t rx_errors;
    uint64_t tx_errors;
    uint64_t rx_crc_errors;
    uint64_t tx_dropped;
    uint64_t rx_dropped;

    uint64_t rx_frame_errors;
    uint64_t rx_length_errors;
    uint64_t rx_alignment_errors;
    uint64_t rx_missed_errors;
    uint64_t rx_no_buffer_count;
    uint64_t rx_frame_check_sequence_errors;

    uint64_t in_bad_octets;
    uint64_t in_fragments;
    uint64_t in_jabber;
    uint64_t in_oversize;
    uint64_t in_undersize;
    uint64_t in_discards;

    uint64_t phy_receive_errors;
    uint64_t phy_serdes_ber_errors;
    uint64_t phy_false_carrier_sense_errors;
    uint64_t phy_local_rcvr_nok;
    uint64_t phy_remote_rcv_nok;
};

/* with this model is fed */
typedef struct cable_features {
    float frame_err_ppm;                // per million RX packets
    float length_err_ppm;               // per million RX packets
    float speed_change_count_10m;       // count of link speed changes in last 10 min
    float speed_is_downgraded;          // 1 if current speed < previous speed, else 0
    float rx_err_ppm;                   // per million RX packets

    float phy_receive_errors_rate;      // phy stats delta / sec
    float phy_serdes_ber_errors_rate;   // phy stats delta / sec
    float fcs_per_million_pkts;         // CRC/FCS delta per million RX packets
    float rx_error_rate;                // RX errors delta / sec
    // float host_rx_crc_rate;          // host RX CRC delta / sec
    // float tx_dropped_rate;           // host TX dropped delta / sec
    // float bad_octets_rate;           // bad_octets delta / sec
    float phy_local_rcvr_nok_rate;      // phy_local_rcvr_nok delta / sec
    float phy_remote_rcv_nok_rate;      // phy_remote_rcv_nok delt / sec
    //float mean_fcs_per_million;       // fcs errors per million packets
    //float max_fcs_per_million;        // max fcs errors per million packets
    float utilization;                  // (rx_bps + tx_bps) / link_bps
    float flaps_10m;                    // carrier transition count in last 10 min
    // float temp_slope_10m;            // (temp_now - temp_10m_ago) / 600
    float temp_delta_10m;               // (temp_now - temp_10m_ago)
} cable_features;

typedef struct cable_history_metrics {
    float speed_change_count_1h;
    float flaps_1h;
    float temp_delta_1h;
} cable_history_metrics;

typedef struct cable_anomaly_history_metrics {
    int suspicious_count_1h;
    int anomalous_count_1h;
} cable_anomaly_history_metrics;

typedef struct raw_sample {
    double ts_sec;  // timestamp

    int carrier;
    int speed_mbps;
    float temp_c;

    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t rx_bytes;
    uint64_t tx_bytes;

    uint64_t rx_errors;
    uint64_t rx_crc_errors;
    uint64_t tx_dropped;

    uint64_t rx_frame_errors;
    uint64_t rx_length_errors;

    uint64_t phy_receive_errors;
    uint64_t phy_serdes_ber_errors;

    //uint64_t bad_octets_rate;
    uint64_t phy_local_rcvr_nok;
    uint64_t phy_remote_rcv_nok;
} raw_sample;

typedef struct cable_feature_state {
    int initialized;
    raw_sample prev;

    rolling_window speed_change_10m;
    rolling_window link_flap_10m;
    rolling_window temp_10m;
    rolling_window speed_change_1h;
    rolling_window link_flap_1h;
    rolling_window temp_1h;
    rolling_window anomalous_1h;
    rolling_window suspicious_1h;
    rolling_window fcs_ppm_10m;
} cable_feature_state;

#endif // CABLE_AUTOENCODER_XNNPACK_H