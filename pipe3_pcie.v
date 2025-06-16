// PCIe Gen3 PIPE Interface Implementation
// Supports 8.0 GT/s data rate with 128b/130b encoding

module pcie_gen3_pipe_interface #(
    parameter PIPE_WIDTH = 32,          // PIPE data width (16/32 bits)
    parameter NUM_LANES = 1,            // Number of PCIe lanes (1, 2, 4, 8, 16)
    parameter ENABLE_EQ = 1             // Enable equalization support
) (
    // Clock and Reset
    input  wire                         pclk,           // PIPE clock (125/250 MHz)
    input  wire                         preset_n,       // PIPE reset (active low)
    
    // PCIe Link Training and Status Machine (LTSSM) Interface
    input  wire [5:0]                   ltssm_state,    // Current LTSSM state
    output reg  [1:0]                   power_down,     // Power management state
    
    // PIPE Transmit Interface
    input  wire [PIPE_WIDTH-1:0]        tx_data,        // Transmit data
    input  wire [(PIPE_WIDTH/8)-1:0]    tx_data_k,      // Transmit data K-char
    input  wire                         tx_elecidle,    // Transmit electrical idle
    input  wire                         tx_compliance,  // Transmit compliance pattern
    input  wire [1:0]                   tx_margin,      // Transmit voltage margin
    input  wire                         tx_swing,       // Transmit voltage swing
    input  wire                         tx_deemph,      // Transmit de-emphasis
    
    // PIPE Receive Interface  
    output reg  [PIPE_WIDTH-1:0]        rx_data,        // Receive data
    output reg  [(PIPE_WIDTH/8)-1:0]    rx_data_k,      // Receive data K-char
    output reg                          rx_valid,       // Receive data valid
    output reg                          rx_elecidle,    // Receive electrical idle
    output reg  [2:0]                   rx_status,      // Receive status
    
    // Gen3 Equalization Interface (conditional)
    input  wire [17:0]                  tx_preset,      // Transmit equalization preset
    input  wire [5:0]                   tx_preset_index,// Transmit preset index
    output reg  [17:0]                  rx_preset,      // Receive equalization preset
    output reg  [5:0]                   rx_preset_index,// Receive preset index
    input  wire                         eq_control,     // Equalization control
    output reg                          eq_status,      // Equalization status
    input  wire [3:0]                   eq_lf,          // Equalization LF
    input  wire [3:0]                   eq_fs,          // Equalization FS
    output reg  [3:0]                   eq_lf_out,      // Equalization LF output
    output reg  [3:0]                   eq_fs_out,      // Equalization FS output
    
    // Gen3-specific signals
    input  wire [1:0]                   rate,           // Data rate (00=2.5GT/s, 01=5GT/s, 10=8GT/s)
    output reg                          phy_status,     // PHY status
    input  wire                         tx_detect_rx,   // Transmit detect receive
    output reg                          rx_polarity,    // Receive polarity inversion
    
    // Block Alignment for Gen3 128b/130b
    output reg                          block_align_status, // Block alignment status
    output reg  [1:0]                   sync_header,    // Sync header for 128b/130b
    
    // Physical Layer Interface (to SerDes)
    output wire                         tx_serial_p,    // Transmit serial data +
    output wire                         tx_serial_n,    // Transmit serial data -
    input  wire                         rx_serial_p,    // Receive serial data +
    input  wire                         rx_serial_n,    // Receive serial data -
    
    // Debug and Test
    output reg  [7:0]                   debug_status,   // Debug status register
    input  wire                         loopback_en     // Internal loopback enable
);

// LTSSM State Definitions
localparam [5:0] LTSSM_DETECT_QUIET     = 6'h00;
localparam [5:0] LTSSM_DETECT_ACT       = 6'h01;
localparam [5:0] LTSSM_POLL_ACTIVE      = 6'h02;
localparam [5:0] LTSSM_POLL_COMPLIANCE  = 6'h03;
localparam [5:0] LTSSM_POLL_CONFIG      = 6'h04;
localparam [5:0] LTSSM_PRE_DETECT_QUIET = 6'h05;
localparam [5:0] LTSSM_DETECT_WAIT      = 6'h06;
localparam [5:0] LTSSM_CFG_LINKWD_START = 6'h07;
localparam [5:0] LTSSM_CFG_LINKWD_ACEPT = 6'h08;
localparam [5:0] LTSSM_CFG_LANENUM_WAIT = 6'h09;
localparam [5:0] LTSSM_CFG_LANENUM_ACEPT = 6'h0A;
localparam [5:0] LTSSM_CFG_COMPLETE     = 6'h0B;
localparam [5:0] LTSSM_CFG_IDLE         = 6'h0C;
localparam [5:0] LTSSM_RCVRY_LOCK       = 6'h0D;
localparam [5:0] LTSSM_RCVRY_SPEED      = 6'h0E;
localparam [5:0] LTSSM_RCVRY_RCVRCFG    = 6'h0F;
localparam [5:0] LTSSM_RCVRY_IDLE       = 6'h10;
localparam [5:0] LTSSM_RCVRY_EQ0        = 6'h11;
localparam [5:0] LTSSM_RCVRY_EQ1        = 6'h12;
localparam [5:0] LTSSM_RCVRY_EQ2        = 6'h13;
localparam [5:0] LTSSM_RCVRY_EQ3        = 6'h14;
localparam [5:0] LTSSM_L0               = 6'h15;
localparam [5:0] LTSSM_L0S              = 6'h16;
localparam [5:0] LTSSM_L1_IDLE          = 6'h17;
localparam [5:0] LTSSM_L2_IDLE          = 6'h18;
localparam [5:0] LTSSM_DISABLED         = 6'h19;
localparam [5:0] LTSSM_LOOPBACK_MASTER  = 6'h1A;
localparam [5:0] LTSSM_LOOPBACK_SLAVE   = 6'h1B;
localparam [5:0] LTSSM_HOT_RESET        = 6'h1C;

// Rate definitions
localparam [1:0] RATE_GEN1 = 2'b00;  // 2.5 GT/s
localparam [1:0] RATE_GEN2 = 2'b01;  // 5.0 GT/s  
localparam [1:0] RATE_GEN3 = 2'b10;  // 8.0 GT/s

// K-character definitions for 8b/10b encoding
localparam [7:0] K28_5 = 8'hBC;  // COM character
localparam [7:0] K23_7 = 8'hF7;  // EIE (Electrical Idle Exit)
localparam [7:0] K27_7 = 8'hFB;  // STP (Start TLP)
localparam [7:0] K29_7 = 8'hFD;  // SDP (Start DLLP)
localparam [7:0] K30_7 = 8'hFE;  // END (End)

// Internal registers and wires
reg [PIPE_WIDTH-1:0]    tx_data_reg;
reg [(PIPE_WIDTH/8)-1:0] tx_data_k_reg;
reg                     rx_elecidle_reg;
reg [2:0]               rx_status_reg;
reg                     block_sync_lock;
reg [6:0]               sync_counter;
reg [1:0]               current_rate;
reg                     equalization_active;
reg [3:0]               eq_phase;

// 128b/130b Block Sync for Gen3
reg [129:0]             rx_block_buffer;
reg [6:0]               block_sync_cnt;
reg                     sync_header_lock;

// Scrambler/Descrambler for Gen3 (LFSR-based)
reg [22:0]              tx_scrambler_state;
reg [22:0]              rx_descrambler_state;
wire [PIPE_WIDTH-1:0]   tx_data_scrambled;
wire [PIPE_WIDTH-1:0]   rx_data_descrambled;

// Clock domain crossing FIFOs (simplified representation)
reg [PIPE_WIDTH+PIPE_WIDTH/8:0] tx_fifo [0:15];
reg [PIPE_WIDTH+PIPE_WIDTH/8:0] rx_fifo [0:15];
reg [3:0]                       tx_fifo_wr_ptr;
reg [3:0]                       tx_fifo_rd_ptr;
reg [3:0]                       rx_fifo_wr_ptr;
reg [3:0]                       rx_fifo_rd_ptr;

// Instantiate SerDes (simplified interface)
serdes_gen3 #(
    .DATA_WIDTH(PIPE_WIDTH),
    .NUM_LANES(NUM_LANES)
) u_serdes (
    .ref_clk(pclk),
    .reset_n(preset_n),
    .rate(current_rate),
    
    // Transmit
    .tx_data(tx_data_scrambled),
    .tx_data_k(tx_data_k_reg),
    .tx_elecidle(tx_elecidle),
    .tx_compliance(tx_compliance),
    .tx_serial_p(tx_serial_p),
    .tx_serial_n(tx_serial_n),
    
    // Receive  
    .rx_serial_p(rx_serial_p),
    .rx_serial_n(rx_serial_n),
    .rx_data(rx_data_descrambled),
    .rx_data_k(rx_data_k),
    .rx_valid(rx_valid),
    .rx_elecidle(rx_elecidle_reg),
    
    // Equalization (Gen3)
    .tx_preset(tx_preset),
    .rx_preset(rx_preset),
    .eq_control(eq_control),
    .eq_status(eq_status),
    .eq_lf(eq_lf),
    .eq_fs(eq_fs),
    .eq_lf_out(eq_lf_out),
    .eq_fs_out(eq_fs_out),
    
    // Status
    .phy_status(phy_status),
    .rx_polarity(rx_polarity)
);

// Data Scrambling for Gen3 (polynomial: x^23 + x^21 + x^16 + x^8 + x^5 + x^2 + x + 1)
always @(posedge pclk or negedge preset_n) begin
    if (!preset_n) begin
        tx_scrambler_state <= 23'h7FFFFF; // Initial seed
    end else if (current_rate == RATE_GEN3) begin
        // Update scrambler LFSR
        tx_scrambler_state <= {tx_scrambler_state[21:0], 
                              tx_scrambler_state[22] ^ tx_scrambler_state[20] ^ 
                              tx_scrambler_state[15] ^ tx_scrambler_state[7] ^ 
                              tx_scrambler_state[4] ^ tx_scrambler_state[1] ^ 
                              tx_scrambler_state[0]};
    end
end

// Generate scrambled data for Gen3
assign tx_data_scrambled = (current_rate == RATE_GEN3) ? 
                          (tx_data_reg ^ tx_scrambler_state[PIPE_WIDTH-1:0]) : 
                          tx_data_reg;

// Data Descrambling for Gen3
always @(posedge pclk or negedge preset_n) begin
    if (!preset_n) begin
        rx_descrambler_state <= 23'h7FFFFF; // Initial seed  
    end else if (current_rate == RATE_GEN3 && rx_valid) begin
        // Update descrambler LFSR
        rx_descrambler_state <= {rx_descrambler_state[21:0],
                                rx_descrambler_state[22] ^ rx_descrambler_state[20] ^ 
                                rx_descrambler_state[15] ^ rx_descrambler_state[7] ^ 
                                rx_descrambler_state[4] ^ rx_descrambler_state[1] ^ 
                                rx_descrambler_state[0]};
    end
end

// Descrambled receive data
always @(posedge pclk) begin
    if (current_rate == RATE_GEN3 && rx_valid) begin
        rx_data <= rx_data_descrambled ^ rx_descrambler_state[PIPE_WIDTH-1:0];
    end else begin
        rx_data <= rx_data_descrambled;
    end
end

// 128b/130b Block Synchronization for Gen3
always @(posedge pclk or negedge preset_n) begin
    if (!preset_n) begin
        block_sync_lock <= 1'b0;
        sync_counter <= 7'b0;
        sync_header_lock <= 1'b0;
        block_sync_cnt <= 7'b0;
        sync_header <= 2'b00;
    end else if (current_rate == RATE_GEN3) begin
        // Look for sync headers (01 or 10 pattern)
        if (rx_valid && !block_sync_lock) begin
            // Check for valid sync header pattern
            if ((rx_data[1:0] == 2'b01) || (rx_data[1:0] == 2'b10)) begin
                if (sync_counter == 7'd64) begin // Found sync header at expected position
                    block_sync_lock <= 1'b1;
                    sync_header_lock <= 1'b1;
                    sync_counter <= 7'b0;
                end else begin
                    sync_counter <= sync_counter + 1'b1;
                end
            end else begin
                sync_counter <= 7'b0; // Reset on invalid pattern
            end
        end else if (block_sync_lock) begin
            // Maintain block sync
            sync_header <= rx_data[1:0];
            if (sync_counter == 7'd129) begin // 130-bit block boundary
                sync_counter <= 7'b0;
                // Verify sync header is still valid
                if ((rx_data[1:0] != 2'b01) && (rx_data[1:0] != 2'b10)) begin
                    block_sync_cnt <= block_sync_cnt + 1'b1;
                    if (block_sync_cnt >= 7'd16) begin // Lost sync
                        block_sync_lock <= 1'b0;
                        sync_header_lock <= 1'b0;
                        block_sync_cnt <= 7'b0;
                    end
                end else begin
                    block_sync_cnt <= 7'b0; // Reset error counter
                end
            end else begin
                sync_counter <= sync_counter + 1'b1;
            end
        end
    end else begin
        // For Gen1/Gen2, block sync is not used
        block_sync_lock <= 1'b1; // Always locked for 8b/10b
        sync_header_lock <= 1'b1;
    end
end

assign block_align_status = block_sync_lock;

// PIPE Interface State Machine
always @(posedge pclk or negedge preset_n) begin
    if (!preset_n) begin
        power_down <= 2'b11;          // P3 state (power down)
        current_rate <= RATE_GEN1;     // Start with Gen1
        equalization_active <= 1'b0;
        eq_phase <= 4'b0;
        tx_data_reg <= {PIPE_WIDTH{1'b0}};
        tx_data_k_reg <= {(PIPE_WIDTH/8){1'b0}};
        rx_status_reg <= 3'b000;
        debug_status <= 8'h00;
    end else begin
        // Update current rate
        current_rate <= rate;
        
        // Register transmit data
        tx_data_reg <= tx_data;
        tx_data_k_reg <= tx_data_k;
        
        // Update receive status
        rx_status <= rx_status_reg;
        rx_elecidle <= rx_elecidle_reg;
        
        case (ltssm_state)
            LTSSM_DETECT_QUIET, LTSSM_DETECT_WAIT: begin
                power_down <= 2'b11;  // P3 state
                debug_status[0] <= 1'b1;
            end
            
            LTSSM_DETECT_ACT: begin
                power_down <= 2'b00;  // P0 state
                debug_status[1] <= 1'b1;
            end
            
            LTSSM_POLL_ACTIVE, LTSSM_POLL_COMPLIANCE, LTSSM_POLL_CONFIG: begin
                power_down <= 2'b00;  // P0 state
                debug_status[2] <= 1'b1;
            end
            
            LTSSM_CFG_LINKWD_START, LTSSM_CFG_LINKWD_ACEPT,
            LTSSM_CFG_LANENUM_WAIT, LTSSM_CFG_LANENUM_ACEPT,
            LTSSM_CFG_COMPLETE, LTSSM_CFG_IDLE: begin
                power_down <= 2'b00;  // P0 state
                debug_status[3] <= 1'b1;
            end
            
            LTSSM_RCVRY_LOCK, LTSSM_RCVRY_SPEED, LTSSM_RCVRY_RCVRCFG, LTSSM_RCVRY_IDLE: begin
                power_down <= 2'b00;  // P0 state
                debug_status[4] <= 1'b1;
            end
            
            // Gen3 Equalization phases
            LTSSM_RCVRY_EQ0, LTSSM_RCVRY_EQ1, LTSSM_RCVRY_EQ2, LTSSM_RCVRY_EQ3: begin
                power_down <= 2'b00;  // P0 state
                equalization_active <= 1'b1;
                case (ltssm_state)
                    LTSSM_RCVRY_EQ0: eq_phase <= 4'h0;
                    LTSSM_RCVRY_EQ1: eq_phase <= 4'h1;
                    LTSSM_RCVRY_EQ2: eq_phase <= 4'h2;
                    LTSSM_RCVRY_EQ3: eq_phase <= 4'h3;
                endcase
                debug_status[5] <= 1'b1;
            end
            
            LTSSM_L0: begin
                power_down <= 2'b00;  // P0 state - Normal operation
                equalization_active <= 1'b0;
                debug_status[6] <= 1'b1;
            end
            
            LTSSM_L0S: begin
                power_down <= 2'b01;  // P0s state
            end
            
            LTSSM_L1_IDLE: begin
                power_down <= 2'b10;  // P1 state
            end
            
            LTSSM_L2_IDLE: begin
                power_down <= 2'b11;  // P2 state
            end
            
            LTSSM_DISABLED: begin
                power_down <= 2'b11;  // P3 state
                debug_status[7] <= 1'b1;
            end
            
            default: begin
                power_down <= 2'b00;  // Default to P0
            end
        endcase
    end
end

// Gen3 Equalization Control
generate
    if (ENABLE_EQ) begin : gen_equalization
        always @(posedge pclk or negedge preset_n) begin
            if (!preset_n) begin
                rx_preset <= 18'h00000;
                rx_preset_index <= 6'h00;
            end else if (equalization_active && (current_rate == RATE_GEN3)) begin
                case (eq_phase)
                    4'h0: begin // Phase 0 - Preset evaluation
                        rx_preset_index <= tx_preset_index;
                        rx_preset <= tx_preset;
                    end
                    4'h1: begin // Phase 1 - Coefficient search
                        // Implement coefficient search algorithm
                        // This is a simplified version
                        if (eq_control) begin
                            rx_preset[5:0] <= rx_preset[5:0] + 1'b1; // Increment C-1
                        end
                    end
                    4'h2: begin // Phase 2 - Coefficient evaluation  
                        // Evaluate current coefficients and provide feedback
                        eq_status <= (rx_preset[5:0] < 6'h20) ? 1'b1 : 1'b0;
                    end
                    4'h3: begin // Phase 3 - Final settings
                        // Apply final equalization settings
                        // Implementation specific
                    end
                endcase
            end
        end
    end else begin : no_equalization
        always @(*) begin
            rx_preset = 18'h00000;
            rx_preset_index = 6'h00;
        end
    end
endgenerate

// Transmit and Receive FIFO Management (Clock Domain Crossing)
always @(posedge pclk or negedge preset_n) begin
    if (!preset_n) begin
        tx_fifo_wr_ptr <= 4'h0;
        tx_fifo_rd_ptr <= 4'h0;
        rx_fifo_wr_ptr <= 4'h0;
        rx_fifo_rd_ptr <= 4'h0;
    end else begin
        // TX FIFO write
        if (tx_data_k != 0 || tx_data != 0) begin
            tx_fifo[tx_fifo_wr_ptr] <= {tx_data_k, tx_data};
            tx_fifo_wr_ptr <= tx_fifo_wr_ptr + 1'b1;
        end
        
        // TX FIFO read
        if (tx_fifo_wr_ptr != tx_fifo_rd_ptr) begin
            tx_fifo_rd_ptr <= tx_fifo_rd_ptr + 1'b1;
        end
        
        // RX FIFO write
        if (rx_valid) begin
            rx_fifo[rx_fifo_wr_ptr] <= {rx_data_k, rx_data};
            rx_fifo_wr_ptr <= rx_fifo_wr_ptr + 1'b1;
        end
        
        // RX FIFO read
        if (rx_fifo_wr_ptr != rx_fifo_rd_ptr) begin
            rx_fifo_rd_ptr <= rx_fifo_rd_ptr + 1'b1;
        end
    end
end

endmodule

// Simplified SerDes Interface Module
module serdes_gen3 #(
    parameter DATA_WIDTH = 32,
    parameter NUM_LANES = 1
) (
    input  wire                     ref_clk,
    input  wire                     reset_n,
    input  wire [1:0]              rate,
    
    // Transmit Interface
    input  wire [DATA_WIDTH-1:0]   tx_data,
    input  wire [(DATA_WIDTH/8)-1:0] tx_data_k,
    input  wire                    tx_elecidle,
    input  wire                    tx_compliance,
    output wire                    tx_serial_p,
    output wire                    tx_serial_n,
    
    // Receive Interface
    input  wire                    rx_serial_p,
    input  wire                    rx_serial_n,
    output reg  [DATA_WIDTH-1:0]   rx_data,
    output reg  [(DATA_WIDTH/8)-1:0] rx_data_k,
    output reg                     rx_valid,
    output reg                     rx_elecidle,
    
    // Gen3 Equalization
    input  wire [17:0]             tx_preset,
    output reg  [17:0]             rx_preset,
    input  wire                    eq_control,
    output reg                     eq_status,
    input  wire [3:0]              eq_lf,
    input  wire [3:0]              eq_fs,
    output reg  [3:0]              eq_lf_out,
    output reg  [3:0]              eq_fs_out,
    
    // Status
    output reg                     phy_status,
    output reg                     rx_polarity
);

// Internal registers for SerDes simulation
reg [DATA_WIDTH-1:0] tx_data_reg;
reg [DATA_WIDTH-1:0] rx_data_internal;
reg tx_serial_p_reg, tx_serial_n_reg;

// Clock generation based on rate
reg pll_lock;
reg [2:0] clk_div_counter;
wire serdes_clk;

// Simple clock divider for different rates
always @(posedge ref_clk or negedge reset_n) begin
    if (!reset_n) begin
        clk_div_counter <= 3'b0;
        pll_lock <= 1'b0;
    end else begin
        clk_div_counter <= clk_div_counter + 1'b1;
        pll_lock <= 1'b1; // Simplified - assume PLL always locks
    end
end

// Generate SerDes clock based on rate
assign serdes_clk = (rate == 2'b00) ? clk_div_counter[2] :      // Gen1: div by 8
                   (rate == 2'b01) ? clk_div_counter[1] :      // Gen2: div by 4  
                   (rate == 2'b10) ? clk_div_counter[0] :      // Gen3: div by 2
                   ref_clk;                                    // Default

// Transmit path
always @(posedge serdes_clk or negedge reset_n) begin
    if (!reset_n) begin
        tx_data_reg <= {DATA_WIDTH{1'b0}};
        tx_serial_p_reg <= 1'b0;
        tx_serial_n_reg <= 1'b1;
    end else begin
        tx_data_reg <= tx_data;
        
        // Simplified serialization (actual implementation would be much more complex)
        if (tx_elecidle) begin
            tx_serial_p_reg <= 1'b0;
            tx_serial_n_reg <= 1'b0;
        end else if (tx_compliance) begin
            // Generate compliance pattern
            tx_serial_p_reg <= clk_div_counter[0];
            tx_serial_n_reg <= ~clk_div_counter[0];
        end else begin
            // Normal data transmission
            tx_serial_p_reg <= tx_data[0];  // Simplified - send LSB
            tx_serial_n_reg <= ~tx_data[0];
        end
    end
end

assign tx_serial_p = tx_serial_p_reg;
assign tx_serial_n = tx_serial_n_reg;

// Receive path  
always @(posedge serdes_clk or negedge reset_n) begin
    if (!reset_n) begin
        rx_data <= {DATA_WIDTH{1'b0}};
        rx_data_k <= {(DATA_WIDTH/8){1'b0}};
        rx_valid <= 1'b0;
        rx_elecidle <= 1'b1;
        phy_status <= 1'b0;
        rx_polarity <= 1'b0;
    end else begin
        // Simplified receive - in real implementation this would include
        // CDR, equalization, 8b/10b or 128b/130b decoding, etc.
        
        // Detect electrical idle
        rx_elecidle <= (rx_serial_p == 1'b0) && (rx_serial_n == 1'b0);
        
        // Simple data recovery (highly simplified)
        if (!rx_elecidle) begin
            rx_data <= {rx_data[DATA_WIDTH-2:0], rx_serial_p};
            rx_valid <= 1'b1;
        end else begin
            rx_valid <= 1'b0;
        end
        
        phy_status <= pll_lock;
        
        // Detect polarity inversion
        rx_polarity <= (rx_serial_p < rx_serial_n);
    end
end

// Gen3 Equalization (simplified)
always @(posedge ref_clk or negedge reset_n) begin
    if (!reset_n) begin
        rx_preset <= 18'h00000;
        eq_status <= 1'b0;
        eq_lf_out <= 4'h0;
        eq_fs_out <= 4'h0;
    end else if (rate == 2'b10) begin // Gen3 only
        rx_preset <= tx_preset;
        eq_status <= eq_control;
        eq_lf_out <= eq_lf;
        eq_fs_out <= eq_fs;
    end
end

endmodule

// Test bench for PCIe Gen3 PIPE Interface
module tb_pcie_gen3_pipe_interface;

reg         pclk;
reg         preset_n;
reg [5:0]   ltssm_state;
reg [31:0]  tx_data;
reg [3:0]   tx_data_k;
reg         tx_elecidle;
reg         tx_compliance;
reg [1:0]   tx_margin;
reg         tx_swing;
reg         tx_deemph;
reg [17:0]  tx_preset;
reg [5:0]   tx_preset_index;
reg         eq_control;
reg [3:0]   eq_lf;
reg [3:0]   eq_fs;
reg [1:0]   rate;
reg         tx_detect_rx;
reg         loopback_en;

wire [1:0]  power_down;
wire [31:0] rx_data;
wire [3:0]  rx_data_k;
wire        rx_valid;
wire        rx_elecidle;
wire [2:0]  rx_status;
wire [17:0] rx_preset;
wire [5:0]  rx_preset_index;
wire        eq_status;
wire [3:0]  eq_lf_out;
wire [3:0]  eq_fs_out;
wire        phy_status;
wire        rx_polarity;
wire        block_align_status;
wire [1:0]  sync_header;
wire        tx_serial_p;
wire        tx_serial_n;
wire [7:0]  debug_status;

// Clock generation
initial begin
    pclk = 0;
    forever #2 pclk = ~pclk; // 250 MHz clock
end

// DUT instantiation
pcie_gen3_pipe_interface #(
    .PIPE_WIDTH(32),
    .NUM_LANES(1),
    .ENABLE_EQ(1)
) dut (
    .pclk(pclk),
    .preset_n(preset_n),
    .ltssm_state(ltssm_state),
    .power_down(power_down),
    
    // Transmit
    .tx_data(tx_data),
    .tx_data_k(tx_data_k),
    .tx_elecidle(tx_elecidle),
    .tx_compliance(tx_compliance),
    .tx_margin(tx_margin),
    .tx_swing(tx_swing),
    .tx_deemph(tx_deemph),
    
    // Receive
    .rx_data(rx_data),
    .rx_data_k(rx_data_k),
    .rx_valid(rx_valid),
    .rx_elecidle(rx_elecidle),
    .rx_status(rx_status),
    
    // Equalization
    .tx_preset(tx_preset),
    .tx_preset_index(tx_preset_index),
    .rx_preset(rx_preset),
    .rx_preset_index(rx_preset_index),
    .eq_control(eq_control),
    .eq_status(eq_status),
    .eq_lf(eq_lf),
    .eq_fs(eq_fs),
    .eq_lf_out(eq_lf_out),
    .eq_fs_out(eq_fs_out),
    
    // Control
    .rate(rate),
    .phy_status(phy_status),
    .tx_detect_rx(tx_detect_rx),
    .rx_polarity(rx_polarity),
    .block_align_status(block_align_status),
    .sync_header(sync_header),
    
    // Physical
    .tx_serial_p(tx_serial_p),
    .tx_serial_n(tx_serial_n),
    .rx_serial_p(tx_serial_p),  // Loopback for test
    .rx_serial_n(tx_serial_n),  // Loopback for test
    
    .debug_status(debug_status),
    .loopback_en(loopback_en)
);

// Test sequence
initial begin
    // Initialize signals
    preset_n = 0;
    ltssm_state = 6'h00; // DETECT_QUIET
    tx_data = 32'h0;
    tx_data_k = 4'h0;
    tx_elecidle = 1;
    tx_compliance = 0;
    tx_margin = 2'b00;
    tx_swing = 0;
    tx_deemph = 0;
    tx_preset = 18'h0;
    tx_preset_index = 6'h0;
    eq_control = 0;
    eq_lf = 4'h0;
    eq_fs = 4'h0;
    rate = 2'b00; // Start with Gen1
    tx_detect_rx = 0;
    loopback_en = 1; // Enable loopback for testing
    
    // Release reset
    #20;
    preset_n = 1;
    
    $display("Starting PCIe Gen3 PIPE Interface Test");
    $display("Time\tLTSSM\t\tRate\tPower\tRX_Valid\tBlock_Sync");
    
    // Test sequence: Detection
    #100;
    ltssm_state = 6'h01; // DETECT_ACT
    tx_elecidle = 0;
    
    // Polling
    #200;
    ltssm_state = 6'h02; // POLL_ACTIVE
    tx_data = 32'hBC_BC_BC_BC; // Send COM characters
    tx_data_k = 4'hF;
    
    // Configuration
    #300;
    ltssm_state = 6'h0B; // CFG_COMPLETE
    tx_data = 32'h12345678;
    tx_data_k = 4'h0;
    
    // L0 - Normal operation at Gen1
    #400;
    ltssm_state = 6'h15; // L0
    rate = 2'b00; // Gen1
    
    // Test data transmission at Gen1
    repeat (10) begin
        #10;
        tx_data = $random;
        tx_data_k = 4'h0;
    end
    
    // Speed change to Gen2
    #500;
    ltssm_state = 6'h0D; // RCVRY_LOCK
    rate = 2'b01; // Gen2
    
    #100;
    ltssm_state = 6'h0E; // RCVRY_SPEED
    
    #100;
    ltssm_state = 6'h15; // L0
    
    // Test data transmission at Gen2
    repeat (10) begin
        #10;
        tx_data = $random;
        tx_data_k = 4'h0;
    end
    
    // Speed change to Gen3 with equalization
    #700;
    ltssm_state = 6'h0D; // RCVRY_LOCK
    rate = 2'b10; // Gen3
    
    #100;
    ltssm_state = 6'h11; // RCVRY_EQ0
    tx_preset = 18'h12345;
    tx_preset_index = 6'h0A;
    eq_control = 1;
    eq_lf = 4'h5;
    eq_fs = 4'h3;
    
    #100;
    ltssm_state = 6'h12; // RCVRY_EQ1
    
    #100;
    ltssm_state = 6'h13; // RCVRY_EQ2
    
    #100;
    ltssm_state = 6'h14; // RCVRY_EQ3
    
    #100;
    ltssm_state = 6'h15; // L0
    eq_control = 0;
    
    // Test data transmission at Gen3
    repeat (20) begin
        #10;
        tx_data = $random;
        tx_data_k = 4'h0;
    end
    
    // Test power management
    #1000;
    ltssm_state = 6'h16; // L0S
    
    #100;
    ltssm_state = 6'h17; // L1_IDLE
    
    #100;
    ltssm_state = 6'h15; // Back to L0
    
    // Test compliance mode
    #1200;
    ltssm_state = 6'h03; // POLL_COMPLIANCE
    tx_compliance = 1;
    
    #200;
    tx_compliance = 0;
    ltssm_state = 6'h15; // L0
    
    // Final data test
    repeat (10) begin
        #10;
        tx_data = $random;
        tx_data_k = 4'h0;
    end
    
    #100;
    $display("Test completed successfully");
    $finish;
end

// Monitor
always @(posedge pclk) begin
    $display("%0t\t%h\t\t%0d\t%0d\t%0d\t\t%0d", 
             $time, ltssm_state, rate, power_down, rx_valid, block_align_status);
end

// Generate VCD file for waveform viewing
initial begin
    $dumpfile("pcie_gen3_pipe.vcd");
    $dumpvars(0, tb_pcie_gen3_pipe_interface);
end

endmodule

// Additional utility modules for PCIe Gen3 PIPE

// 8b/10b Encoder for Gen1/Gen2
module encoder_8b10b (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] data_in,
    input  wire       k_char,
    output reg  [9:0] data_out,
    output reg        disparity
);

// 8b/10b encoding tables (simplified)
reg [9:0] encode_table [0:255];
reg [9:0] k_char_table [0:255];
reg running_disparity;

initial begin
    // Initialize encoding tables (simplified - real implementation needs full tables)
    encode_table[8'h00] = 10'b0110001011; // D0.0
    encode_table[8'h01] = 10'b1001110100; // D1.0
    // ... (full table would have 256 entries)
    
    k_char_table[8'hBC] = 10'b0011111010; // K28.5 (COM)
    k_char_table[8'hF7] = 10'b1110101000; // K23.7 (EIE)
    // ... (K character table)
end

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        data_out <= 10'b0;
        running_disparity <= 1'b0;
        disparity <= 1'b0;
    end else begin
        if (k_char) begin
            data_out <= k_char_table[data_in];
        end else begin
            data_out <= encode_table[data_in];
        end
        
        // Update running disparity (simplified)
        running_disparity <= running_disparity ^ (^data_out);
        disparity <= running_disparity;
    end
end

endmodule

// 128b/130b Encoder for Gen3
module encoder_128b130b (
    input  wire         clk,
    input  wire         reset_n,
    input  wire [127:0] data_in,
    input  wire         sync_header,
    output reg  [129:0] data_out
);

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        data_out <= 130'b0;
    end else begin
        // Add sync header based on block type
        if (sync_header) begin
            data_out <= {2'b01, data_in}; // Data block
        end else begin
            data_out <= {2'b10, data_in}; // Control block
        end
    end
end

endmodule

// Clock Data Recovery (CDR) module
module cdr_gen3 (
    input  wire       ref_clk,
    input  wire       reset_n,
    input  wire       serial_data,
    input  wire [1:0] rate,
    output reg        recovered_clk,
    output reg        lock_status
);

reg [7:0] phase_counter;
reg [3:0] lock_counter;
reg       edge_detected;
reg       serial_data_reg;

always @(posedge ref_clk or negedge reset_n) begin
    if (!reset_n) begin
        phase_counter <= 8'b0;
        lock_counter <= 4'b0;
        lock_status <= 1'b0;
        recovered_clk <= 1'b0;
        edge_detected <= 1'b0;
        serial_data_reg <= 1'b0;
    end else begin
        serial_data_reg <= serial_data;
        edge_detected <= serial_data ^ serial_data_reg;
        
        // Simple phase tracking (real CDR would be much more complex)
        if (edge_detected) begin
            if (lock_counter < 4'hF) begin
                lock_counter <= lock_counter + 1'b1;
            end else begin
                lock_status <= 1'b1;
            end
            phase_counter <= 8'h00; // Reset phase on edge
        end else begin
            phase_counter <= phase_counter + 1'b1;
        end
        
        // Generate recovered clock based on rate
        case (rate)
            2'b00: recovered_clk <= phase_counter[3]; // Gen1
            2'b01: recovered_clk <= phase_counter[2]; // Gen2
            2'b10: recovered_clk <= phase_counter[1]; // Gen3
            default: recovered_clk <= phase_counter[2];
        endcase
    end
end

endmodule