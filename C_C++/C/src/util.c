#include <arpa/inet.h>
#include <json-c/json.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "utlist.h"
#include "util.h"

#define IPFIX_VERSION   10

// IPFIX message header is fixed 16 bytes (network byte order)
struct ipfix_hdr {
    uint16_t version;
    uint16_t length;
    uint32_t export_time;
    uint32_t seq;
    uint32_t odid;
};

static uint16_t rd16(const unsigned char *p)
{
    uint16_t v;
    memcpy(&v,p,2);

    return ntohs(v);
}

static uint32_t rd32(const unsigned char *p)
{
    uint32_t v;
    memcpy(&v,p,4);

    return ntohl(v);
};

static uint64_t rd64(const unsigned char *p)
{
    return ((uint64_t)p[0] << 56) |
           ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) |
           ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) |
           ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8)  |
           ((uint64_t)p[7]);
}

static uint64_t rd_uint(const unsigned char *p, uint16_t len)
{
    switch (len) {
        case 1: return p[0];
        case 2: return rd16(p);
        case 4: return rd32(p);
        case 8: return rd64(p);
        default: return 0;
    }
}

void clear_set_list(struct set *set_list)
{
    struct set *set_node, *set_node_tmp;
    DL_FOREACH_SAFE(set_list, set_node, set_node_tmp) {
        struct record_node *record_node, *record_node_tmp;
        DL_FOREACH_SAFE(set_node->record_list, record_node, record_node_tmp) {
            DL_DELETE(set_node->record_list, record_node);
            free(record_node);
        }
        DL_DELETE(set_list, set_node);
        free(set_node);
    }
}

static struct ipfix_template templates[MAX_TEMPLATES];
static size_t template_count = 0;

static struct ipfix_template *find_template(uint32_t odid, uint16_t template_id)
{
    for (size_t i = 0; i < template_count; ++i) {
        if (templates[i].odid == odid && templates[i].template_id == template_id) {
            return &templates[i];
        }
    }
    return NULL;
}

static struct ipfix_template *upsert_template(uint32_t odid, uint16_t template_id)
{
    struct ipfix_template *tpl = find_template(odid, template_id);
    if (tpl) {
        tpl->field_count = 0;
        return tpl;
    }

    if (template_count >= MAX_TEMPLATES) {
        return NULL;
    }
    tpl = &templates[template_count++];
    memset(tpl, 0, sizeof(*tpl));
    tpl->odid = odid;
    tpl->template_id = template_id;
    return tpl;
}

static void parse_record_1024(uint16_t id, const unsigned char *p, uint16_t len, struct flow_record_1024 *rec)
{
    uint64_t v = rd_uint(p, len);
    switch (id) {
        case IE_SOURCE_IPV4_ADDRESS:
            rec->sourceIPv4Address = v;
            break;
        case IE_DESTINATION_IPV4_ADDRESS:
            rec->destinationIPv4Address = v;
            break;
        case IE_OCTET_DELTA_COUNT:
            rec->octetDeltaCount = v;
            break;
        case IE_PACKET_DELTA_COUNT:
            rec->packetDeltaCount = v;
            break;
        case IE_FLOW_START_SYS_UP_TIME:
            rec->flowStartSysUpTime = (uint32_t)v;
            break;
        case IE_FLOW_END_SYS_UP_TIME:
            rec->flowEndSysUpTime = (uint32_t)v;
            break;
        case IE_DESTINATION_TRANSPORT_PORT:
            rec->destinationTransportPort = (uint16_t)v;
            break;
        case IE_SOURCE_TRANSPORT_PORT:
            rec->sourceTransportPort = (uint16_t)v;
            break;
        case IE_PROTOCOL_IDENTIFIER:
            rec->protocolIdentifier = (uint8_t)v;
            break;
        case IE_IP_CLASS_OF_SERVICE:
            rec->ipClassOfService = (uint8_t)v;
            break;
        case IE_IP_VERSION:
            rec->ipVersion = (uint8_t)v;
            break;
        case IE_FLOW_END_REASON:
            rec->flowEndReason = (uint8_t)v;
            break;
        case IE_FLOW_DIRECTION:
            rec->flowDirection = (uint8_t)v;
            break;
        case IE_TCP_CONTROL_BITS:
            rec->tcpControlBits = (uint8_t)v;
            break;
        case IE_INGRESS_INTERFACE:
            rec->ingressInterface = (uint32_t)v;
            break;
        case IE_EGRESS_INTERFACE:
            rec->egressInterface = (uint32_t)v;
            break;
        case IE_SOURCE_MAC_ADDRESS:
            memcpy(rec->sourceMacAddress, p, 6);
            break;
        case IE_POST_DESTINATION_MAC_ADDRESS:
            memcpy(rec->destinationMacAddress, p, 6);
            break;
        default:
            break;
    }
}

static void parse_template_set(const unsigned char *set, size_t set_data_len, uint32_t odid)
{
    size_t off = 0;
    while (off + 4 <= set_data_len) {
        uint16_t template_id = rd16(set + off);
        uint16_t field_count = rd16(set + off + 2);
        off += 4;

        struct ipfix_template *tpl = upsert_template(odid, template_id);
        uint16_t stored = 0;

        for (uint16_t i = 0; i < field_count; ++i) {
            if (off + 4 > set_data_len) {
                return;
            }
            uint16_t field_id = rd16(set + off);
            uint16_t field_len = rd16(set + off + 2);
            off += 4;

            uint32_t enterprise = 0;
            if (field_id & 0x8000) {
                field_id &= 0x7FFF;
                if (off + 4 > set_data_len) {
                    return;
                }
                enterprise = rd32(set + off);
                off += 4;
            }

            if (tpl && stored < MAX_FIELDS) {
                tpl->fields[stored].id = field_id;
                tpl->fields[stored].length = field_len;
                tpl->fields[stored].enterprise = enterprise;
                stored++;
            }
        }

        if (tpl) {
            tpl->field_count = stored;
        }
    }
}

static void parse_data_set( const unsigned char *set, 
                            size_t set_data_len, 
                            uint32_t odid, 
                            uint16_t template_id, 
                            struct set **set_list)
{
    struct ipfix_template *tpl = find_template(odid, template_id);
    if (!tpl || tpl->field_count == 0) {
        return;
    }

    struct set *set_node = malloc(sizeof(struct set));
    if(!set_node)
        return;

    memset(set_node, 0, sizeof(struct set));
    set_node->record_list = NULL;

    size_t off = 0;
    while (off < set_data_len) {
        size_t rec_start = off;
        struct flow_record rec;
        memset(&rec, 0, sizeof(rec));

        struct record_node *record_node = malloc(sizeof(struct record_node));
        if(!record_node){
            clear_set_list(set_node);
            return;
        }

        memset(record_node, 0, sizeof(struct record_node));
        record_node->record.record_id = template_id;

        for (uint16_t i = 0; i < tpl->field_count; ++i) {
            if (off >= set_data_len) {
                clear_set_list(set_node);
                free(record_node);
                return;
            }

            uint16_t len = tpl->fields[i].length;
            if (len == 0xFFFF) {
                if (off >= set_data_len) {
                    clear_set_list(set_node);
                    free(record_node);
                    return;
                }
                uint8_t l1 = set[off++];
                if (l1 < 255) {
                    len = l1;
                } else {
                    if (off + 2 > set_data_len) {
                        clear_set_list(set_node);
                        free(record_node);
                        return;
                    }
                    len = ((uint16_t)set[off] << 8) | set[off + 1];
                    off += 2;
                }
            }

            if (off + len > set_data_len) {
                clear_set_list(set_node);
                free(record_node);
                return;
            }

            if (tpl->fields[i].enterprise == 0) {
                // parse_field_value(tpl->fields[i].id, set + off, len, &rec);
                if (record_node->record.record_id == RECORD_ID_1024)
                    parse_record_1024(tpl->fields[i].id, set + off, len, &record_node->record.flow_1024);
                // else if(flow->record_id == RECORD_ID_1025)
                //     parse_record_1025(tpl->fields[i].id, set + off, len, &rec);
            }
            off += len;
        }

        // emit_record(&rec);

        if (off == rec_start) {
            free(record_node);
            break;
        }

        if (set_data_len - off < 4) {
            free(record_node);
            break;
        }
        DL_APPEND(set_node->record_list, record_node);
        set_node->count++;
    }
    DL_APPEND(*set_list, set_node);
    (*set_list)->count++;
    printf("%s:%d new SET with %d id=1024 records \n", __FUNCTION__, __LINE__, set_node->count);
}

void parse_ipfix_manually(const unsigned char *buf, size_t n, struct set **set_list)
{
    if (n < 16) return;

    // read header
    uint16_t version = rd16(buf+0);
    uint16_t msg_len = rd16(buf+2);
    //uint32_t export_time = rd32(buf+4);
    //uint32_t seq = rd32(buf+8);
    uint32_t observation_domain_id = rd32(buf+12);

    if (version != 10 || msg_len > n)
        return;

    size_t off = 16;
    while (off + 4 <= msg_len) {
        uint16_t set_id = rd16(buf+off);
        uint16_t set_len = rd16(buf+off+2);
        if (set_len < 4 || off + set_len > msg_len) break;

        const unsigned char *set = buf + off + 4;
        size_t set_data_len = set_len - 4;

        if (set_id == IPFIX_SET_TEMPLATE) {
            parse_template_set(set, set_data_len, observation_domain_id);
        } else if (set_id == IPFIX_SET_OPTIONS_TEMPLATE) {
            // options templates not used for flow parsing here
        } else if (set_id >= 256) {
            printf("Received template id %d \n", set_id);
            // interested in template ids = 1024
            if(set_id == 1024) {
                parse_data_set(set, set_data_len, observation_domain_id, set_id, set_list);
            }
        }

        off += set_len;
    }
}



static void free_vocab(vocab_t *v) {
    if (!v) return;
    free(v->vals);
    v->vals = NULL;
    v->n = 0;
}

void free_vocabs(vocabs_t *v) {
    if (!v) return;
    for (size_t i = 0; i < v->categorical_len; ++i) free_vocab(&v->categorical[i]);
    for (size_t i = 0; i < v->lcf1_len; ++i) free_vocab(&v->lcf1[i]);
    for (size_t i = 0; i < v->lcf2_len; ++i) free_vocab(&v->lcf2[i]);
    free(v->categorical); v->categorical = NULL; v->categorical_len = 0;
    free(v->lcf1); v->lcf1 = NULL; v->lcf1_len = 0;
    free(v->lcf2); v->lcf2 = NULL; v->lcf2_len = 0;
}

static bool load_vocab_group(struct json_object *root,
                             const char *key,
                             vocab_t **out_arr,
                             size_t *out_len) {
    struct json_object *group = NULL;
    if (!json_object_object_get_ex(root, key, &group) ||
        !json_object_is_type(group, json_type_array)) {
        return false;
    }

    size_t g_len = json_object_array_length(group);
    vocab_t *arr = calloc(g_len, sizeof(vocab_t));
    if (!arr) return false;

    for (size_t i = 0; i < g_len; ++i) {
        struct json_object *sub = json_object_array_get_idx(group, i);
        if (!sub || !json_object_is_type(sub, json_type_array)) {
            free(arr);
            return false;
        }
        size_t n = json_object_array_length(sub);
        arr[i].vals = calloc(n, sizeof(double));
        if (!arr[i].vals) { free(arr); return false; }
        arr[i].n = n;

        for (size_t j = 0; j < n; ++j) {
            struct json_object *val = json_object_array_get_idx(sub, j);
            arr[i].vals[j] = json_object_get_double(val);
        }
    }

    *out_arr = arr;
    *out_len = g_len;

    return true;
}

static bool load_float_array(struct json_object *root, const char *key, float *out, int n) {
    struct json_object *arr = NULL;
    if (!json_object_object_get_ex(root, key, &arr)) 
        return false;
    if (!json_object_is_type(arr, json_type_array)) 
        return false;
    if (json_object_array_length(arr) != (size_t)n) 
        return false;

    for (int i = 0; i < n; ++i) {
        struct json_object *item = json_object_array_get_idx(arr, i);
        if (!json_object_is_type(item, json_type_double) &&
            !json_object_is_type(item, json_type_int)) 
            return false;

        out[i] = (float)json_object_get_double(item);
    }
    return true;
}

static bool load_float_key(struct json_object *root, const char *key, float *out) {
    struct json_object *val = NULL;
    if (!json_object_object_get_ex(root, key, &val))
        return false;
    if (!json_object_is_type(val, json_type_double) &&
        !json_object_is_type(val, json_type_int))
        return false;

    *out = (float)json_object_get_double(val);

    return true;
}

bool load_vocabs(const char *path, vocabs_t *out) {
    struct json_object *thresholds = NULL;
    struct json_object *root = NULL;

    memset(out, 0, sizeof(*out));
    root = json_object_from_file(path);

    if (!root)
        return false;

    // load groups
    bool ok = load_vocab_group(root, "categorical", &out->categorical, &out->categorical_len) &&
              load_vocab_group(root, "low_cardinality1", &out->lcf1, &out->lcf1_len) &&
              load_vocab_group(root, "low_cardinality2", &out->lcf2, &out->lcf2_len);

    // Optional cont stats
    out->has_cont_stats = load_float_array(root, "cont_mean", out->cont_mean, 7) &&
                          load_float_array(root, "cont_std",  out->cont_std,  7);

    // load anomaly thresholds
    if (json_object_object_get_ex(root, "anomaly_thresholds", &thresholds) &&
        json_object_is_type(thresholds, json_type_object)) {
        load_float_key(thresholds, "normal",  &out->anomaly_normal);
        load_float_key(thresholds, "suspicious",  &out->anomaly_suspicious);
        load_float_key(thresholds, "anomalous", &out->anomaly_anomalous);
    }
    json_object_put(root);
    if (!ok)
        free_vocabs(out);

    return ok;
}

// returns index+1 if found, else 0 (OOV)
int32_t vocab_index(const vocab_t *v, double raw) {
    if (!v) return 0;
    for (size_t i = 0; i < v->n; ++i) {
        if (v->vals[i] == raw) return (int32_t)(i + 1);
    }
    return 0;
}


/* LOSS FUNCTIONS */
static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

float mean_squared_error(const float *pred, const float *target, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = pred[i] - target[i];
        sum += d * d;
    }
    return sum / (float)n;
}

float binary_cross_entropy(const float *pred, const float *target, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float p = clampf(pred[i], 1e-7f, 1.0f - 1e-7f);
        sum += -(target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p));
    }
    return sum; // or sum / (float)n if you want mean BCE
}


float sparse_cross_entropy(const float *prob, int idx, int n) {
    if (idx < 0 || idx >= n) {
        // treat as OOV -> use tiny prob to avoid log(0)
        return -logf(1e-7f);
    }
    float p = clampf(prob[idx], 1e-7f, 1.0f);
    return -logf(p);
}

int read_config(const char* cfg_path, struct model_config *value)
{
    struct json_object *model_obj, *vocabs_obj, *model_name, *xnnpack_num_threads,
                       *fallback_num_threads, *enabled_acceleration, *threshold;

    struct json_object* root = json_object_from_file(cfg_path);
    if (!root) {
        fprintf(stderr, "Failed to read %s\n", cfg_path);
        return 1;
    }

    if (!json_object_object_get_ex(root, "model_path", &model_obj)) {
        fprintf(stderr, "Missing key: model_path\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "vocabs_path", &vocabs_obj)) {
        fprintf(stderr, "Missing key: vocabs_path\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "model_name", &model_name)) {
        fprintf(stderr, "Missing key: model_name\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "enabled_acceleration", &enabled_acceleration)) {
        fprintf(stderr, "Missing key: enabled_acceleration\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "xnnpack_num_threads", &xnnpack_num_threads)) {
        fprintf(stderr, "Missing key: xnnpack_num_threads\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "fallback_num_threads", &fallback_num_threads)) {
        fprintf(stderr, "Missing key: fallback_num_threads\n");
        json_object_put(root);
        return 1;
    }
    if (!json_object_object_get_ex(root, "threshold", &threshold)) {
        fprintf(stderr, "Missing key: threshold\n");
        json_object_put(root);
        return 1;
    }

    strncpy(value->model_path, json_object_get_string(model_obj), sizeof(value->model_path)-1);
    strncpy(value->vocab_path, json_object_get_string(vocabs_obj), sizeof(value->vocab_path)-1);
    strncpy(value->model_name, json_object_get_string(model_name), sizeof(value->model_name)-1);
    value->enabled_acceleration = json_object_get_int(enabled_acceleration);
    value->xnnpack_num_threads = json_object_get_int(xnnpack_num_threads);
    value->fallback_num_threads = json_object_get_int(fallback_num_threads);
    value->threshold = (float)json_object_get_double(threshold);

    json_object_put(root); // free JSON

    return 0;
}
