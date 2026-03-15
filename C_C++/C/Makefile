CC ?= gcc
STD ?= -std=gnu2x
CFLAGS ?= -O0 -DDEBUG -g3 -fno-omit-frame-pointer -Wall -Wextra $(STD)
# more optimized
# CFLAGS ?= -O3 -DNDEBUG -ffunction-sections -fdata-sections -Wall -Wextra $(STD)

# Buildroot passes these from ipfix_autoencoder.mk
LITERT_PATH ?= /workdir/LiteRT
LITERT_LIB  ?= $(LITERT_PATH)/output/tensorflowlite_c-build/k8-opt/bin/tflite/c
LITERT_INC  ?= $(LITERT_PATH)/tflite/lite/c

TARGET = cable_autoencoder_xnnpack
SRC_FILE = $(TARGET).c
TARGET_BIN = $(TARGET)/$(TARGET)

INC_PATH = -I./* -I./inc -I$(LITERT_PATH) -I$(LITERT_INC)
INC_SRC  = $(wildcard ./src/*.c)
INC_HDR  = $(wildcard ./inc/*.h)
SRC_HDR  = $(wildcard ./src/*.h)

LDFLAGS += -L$(LITERT_LIB) -ljson-c -ltensorflowlite_c -lm

all: $(TARGET_BIN)

$(TARGET):
	mkdir -pv $@

$(TARGET_XNN_PACK):
	mkdir -pv $@

$(TARGET_BIN): ./$(SRC_FILE) $(INC_SRC) $(INC_HDR) $(SRC_HDR) Makefile | $(TARGET)
	$(CC) $(CFLAGS) ./$(SRC_FILE) $(INC_SRC) $(INC_PATH) -o $@ $(LDFLAGS)
	cp $(LITERT_PATH)/output/tensorflowlite_c-build/k8-opt/bin/tflite/c/libtensorflowlite_c.so /usr/lib

clean:
	rm -rf $(TARGET)
	rm -rf $(TARGET_XNN_PACK)

.PHONY: all clean