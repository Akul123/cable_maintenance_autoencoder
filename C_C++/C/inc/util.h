#define RECORD_ID_1024 1024
#define RECORD_ID_1025 1025
#define RECORD_ID_2048 2048
#define RECORD_ID_2049 2049

#define IPFIX_SET_TEMPLATE            2
#define IPFIX_SET_OPTIONS_TEMPLATE    3

#define MAX_TEMPLATES 64
#define MAX_FIELDS    64

#define IE_OCTET_DELTA_COUNT                1
#define IE_PACKET_DELTA_COUNT               2
#define IE_PROTOCOL_IDENTIFIER              4
#define IE_IP_CLASS_OF_SERVICE        		5
#define IE_TCP_CONTROL_BITS           		6
#define IE_SOURCE_TRANSPORT_PORT     	 	7
#define IE_SOURCE_IPV4_ADDRESS        		8
#define IE_INGRESS_INTERFACE          		10
#define IE_DESTINATION_TRANSPORT_PORT 		11
#define IE_DESTINATION_IPV4_ADDRESS   		12
#define IE_EGRESS_INTERFACE           		14
#define IE_FLOW_END_SYS_UP_TIME       		21
#define IE_FLOW_START_SYS_UP_TIME     		22
#define IE_ICMP_TYPE_CODE_IPV4        		32
#define IE_SOURCE_MAC_ADDRESS         		56
#define	IE_POST_DESTINATION_MAC_ADDRESS   	57
#define IE_VLAN_ID                    		58
#define IE_POST_VLAN_ID               		59
#define IE_IP_VERSION                 		60
#define IE_FLOW_DIRECTION             		61
#define IE_DESTINATION_MAC_ADDRESS    		80
#define IE_FLOW_END_REASON            		136

// TCP/UDP flows
struct flow_record_1024 {
    uint32_t sourceIPv4Address;
    uint32_t destinationIPv4Address;
    uint32_t flowStartSysUpTime;
    uint32_t flowEndSysUpTime;
    uint32_t octetDeltaCount;   // BYTES (len 4)
    uint32_t packetDeltaCount;  // PKTS  (len 4)
    uint32_t ingressInterface;  // INPUT_SNMP
    uint32_t egressInterface;   // OUTPUT_SNMP
    uint8_t  flowDirection;     // DIRECTION
    uint8_t  flowEndReason;     // flowEndReason
    uint16_t sourceTransportPort;      // L4_SRC_PORT
    uint16_t destinationTransportPort; // L4_DST_PORT
    uint8_t  protocolIdentifier; // PROTOCOL
    uint8_t  tcpControlBits;     // TCP_FLAGS
    uint8_t  ipVersion;          // IP_PROTOCOL_VERSION
    uint8_t  ipClassOfService;   // IP_TOS
    uint16_t vlanId;             // SRC_VLAN
    uint16_t postVlanId;         // DST_VLAN
    uint8_t  sourceMacAddress[6];       // SRC_MAC
    uint8_t  destinationMacAddress[6];  // DST_MAC
};

// ICMP flows
struct flow_record_1025 {
    uint32_t sourceIPv4Address;
    uint32_t destinationIPv4Address;
    uint32_t flowStartSysUpTime;
    uint32_t flowEndSysUpTime;
    uint32_t octetDeltaCount;   // BYTES (len 4)
    uint32_t packetDeltaCount;  // PKTS  (len 4)
    uint32_t ingressInterface;  // INPUT_SNMP
    uint32_t egressInterface;   // OUTPUT_SNMP
    uint8_t  flowDirection;     // DIRECTION
    uint8_t  flowEndReason;     // flowEndReason
    uint16_t icmpTypeCodeIPv4;  // ICMP_TYPE (type+code)
    uint8_t  protocolIdentifier; // PROTOCOL
    uint8_t  ipVersion;          // IP_PROTOCOL_VERSION
    uint8_t  ipClassOfService;   // IP_TOS
    uint16_t vlanId;             // SRC_VLAN
    uint16_t postVlanId;         // DST_VLAN
    uint8_t  sourceMacAddress[6];       // SRC_MAC
    uint8_t  destinationMacAddress[6];  // DST_MAC
};

enum flow_flags {
    F_OCTETS            = 1u << 0,
    F_PACKETS           = 1u << 1,
    F_FLOW_START        = 1u << 2,
    F_FLOW_END          = 1u << 3,
    F_DST_PORT          = 1u << 4,
    F_SRC_PORT          = 1u << 5,
    F_PROTOCOL          = 1u << 6,
    F_TOS               = 1u << 7,
    F_IP_VERSION        = 1u << 8,
    F_FLOW_END_REASON   = 1u << 9,
    F_FLOW_DIR          = 1u << 10,
    F_TCP_FLAGS         = 1u << 11,
    F_INGRESS_IF        = 1u << 12,
    F_EGRESS_IF         = 1u << 13,
    F_ICMP4             = 1u << 14,
    F_SRC_IP            = 1u << 15,
    F_DST_IP            = 1u << 16,
    F_VLAN              = 1u << 17,
    F_POST_VLAN         = 1u << 18,
    F_SRC_MAC           = 1u << 19,
    F_DST_MAC           = 1u << 20,
};

struct model_config {
    char model_path[256];
    char vocab_path[256];
    char model_name[64];
    float threshold;
    bool enabled_acceleration;
    int xnnpack_num_threads;
    int fallback_num_threads;
};

struct ipfix_field_spec {
    uint16_t id;
    uint16_t length;
    uint32_t enterprise;
};

struct ipfix_template {
    uint32_t odid;
    uint16_t template_id;
    uint16_t field_count;
    struct ipfix_field_spec fields[MAX_FIELDS];
};

struct ipfix_header {
    uint16_t version;
    uint16_t length;
    uint32_t export_time;
    uint32_t seq;
    uint32_t odid; //observation_domain_id
};

struct flow_record {
    union {
        struct flow_record_1024 flow_1024;
        struct flow_record_1025 flow_1025;
    };
    size_t record_id;
};

// node type
struct record_node {
    struct flow_record record;
    unsigned count;
    struct record_node *prev;
    struct record_node *next;
};

struct set {
    struct record_node *record_list;
    unsigned count;
    struct set *prev;
    struct set *next;
};

// vocabs
typedef struct {
    size_t n;
    double *vals; // store as double for easy compare
} vocab_t;

typedef struct {
    vocab_t *categorical;      // tcpControlBits
    size_t categorical_len;
    vocab_t *lcf1;             // protocolIdentifier, ipClassOfService, ipVersion, flowEndReason
    size_t lcf1_len;
    vocab_t *lcf2;             // ingressInterface, egressInterface, icmpTypeCodeIPv4, icmpTypeCodeIPv6
    size_t lcf2_len;
    float cont_mean[7];
    float cont_std[7];
    int has_cont_stats;
    float anomaly_normal;
    float anomaly_suspicious;
    float anomaly_anomalous;
} vocabs_t;

void init_fixbuf(void);
void decode_ipfix_buf(uint8_t *buf, size_t len);
void parse_ipfix_manually(const unsigned char *buf, size_t n, struct set **flows_list);
void clear_set_list(struct set *set_list);
int32_t vocab_index(const vocab_t *v, double raw);
bool load_vocabs(const char *path, vocabs_t *out);
void free_vocabs(vocabs_t *v);

/* LOSS functions */
float mean_squared_error(const float *pred, const float *target, int n);
float binary_cross_entropy(const float *pred, const float *target, size_t n);
float sparse_cross_entropy(const float *prob, int idx, int n);

int read_config(const char* cfg_path, struct model_config *value);
