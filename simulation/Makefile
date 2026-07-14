CORE ?= riscv.CoreExtMem

# Only applicable for CORE=riscv.CoreExtMem:
PIPELINE ?= Static
ISA ?= RV32I
# To instead build with a custom config file, define CONFIG=<path>

BASE_DIR ?= ../core
BUILD_DIR = build
TOPLEVEL_MODULE = Core
VERILOG_FILE_NAME = $(TOPLEVEL_MODULE).v
VERILOG_INPUT = $(BUILD_DIR)/$(VERILOG_FILE_NAME)
VERILATOR_NAME = V$(TOPLEVEL_MODULE)
EXE_NAME ?= sim
EXE_FILE = $(BUILD_DIR)/$(EXE_NAME)

# Enable caching of previous compilations
export OBJCACHE = ccache

# DEV=1: Fast compilation (-O0), file splitting for parallel debug compiles
# DEV=0: Release build (-O3, flat file, maximum simulation speed)
DEV ?= 0

ifeq ($(DEV), 1)
	VERILATOR_OPT = -O0
	CC_OPT = -O0
	SPLIT_FLAGS = --output-split 10000 --output-split-cfuncs 500
else
	VERILATOR_OPT = -O3
	CC_OPT = -O3
	SPLIT_FLAGS =
endif

CFLAGS = $(CC_OPT) -std=c++20 $(EXTRA_CFLAGS)

VERILATOR_FLAGS = --cc --exe
VERILATOR_FLAGS += --trace-fst $(VERILATOR_OPT) $(SPLIT_FLAGS)
VERILATOR_FLAGS += -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CMPCONST -Wno-UNSIGNED

all: $(EXE_FILE)

$(EXE_FILE): $(VERILOG_INPUT) main.cpp
	verilator --build -j $$(nproc) $(VERILATOR_FLAGS) -CFLAGS "$(CFLAGS)" -Mdir $(BUILD_DIR) $(VERILOG_INPUT) main.cpp -o $(EXE_NAME)

$(VERILOG_INPUT): $(BASE_DIR)/$(VERILOG_FILE_NAME)
	mkdir -p $(@D)
	cp $< $@

$(BASE_DIR)/$(VERILOG_FILE_NAME): FORCE
ifeq ($(CORE),riscv.CoreExtMem)
ifdef CONFIG
	cd $(BASE_DIR); sbt "runMain $(CORE) $(abspath $(CONFIG))"
else
	cd $(BASE_DIR); sbt "runMain $(CORE) $(PIPELINE) $(ISA)"
endif
else
	cd $(BASE_DIR); sbt "runMain $(CORE)"
endif

FORCE:

clean:
	rm -rf $(BUILD_DIR)

clean-cache
    ccache -C