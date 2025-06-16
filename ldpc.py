// Low Power LDPC Decoder Implementation
// Features: Early termination, dynamic clock gating, precision reduction
// Architecture: Layered Min-Sum with offset correction

`timescale 1ns / 1ps

// Top-level LDPC Decoder Module
module ldpc_decoder_low_power #(
    parameter BLOCK_SIZE = 672,        // Code block size
    parameter CHECK_NODES = 336,       // Number of check nodes
    parameter MAX_ITER = 20,           // Maximum iterations
    parameter QC_SIZE = 42,            // Quasi-cyclic size
    parameter LLR_WIDTH = 6,           // LLR bit width (reduced for power)
    parameter MIN_SUM_OFFSET = 1,      // Offset for min-sum algorithm
    parameter EARLY_TERM_THRESH = 10   // Early termination threshold
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Input LLRs
    input wire signed [LLR_WIDTH-1:0] llr_in [0:BLOCK_SIZE-1],
    input wire llr_valid,
    
    // Output decoded bits
    output reg [BLOCK_SIZE-1:0] decoded_bits,
    output reg decode_done,
    output reg decode_valid,
    output reg [4:0] iterations_used,
    
    // Power control signals
    input wire low_power_mode,
    input wire [2:0] power_level,  // 0: full power, 7: minimum power
    output reg decoder_idle
);

    // Internal states
    localparam IDLE = 3'b000;
    localparam LOAD = 3'b001;
    localparam DECODE = 3'b010;
    localparam CHECK = 3'b011;
    localparam OUTPUT = 3'b100;
    
    reg [2:0] state, next_state;
    
    // Memory arrays with power gating regions
    reg signed [LLR_WIDTH-1:0] variable_nodes [0:BLOCK_SIZE-1];
    reg signed [LLR_WIDTH-1:0] check_to_variable [0:CHECK_NODES-1][0:5]; // Assuming degree 6
    reg signed [LLR_WIDTH-1:0] variable_to_check [0:BLOCK_SIZE-1][0:2];  // Assuming degree 3
    
    // Syndrome check registers
    reg [CHECK_NODES-1:0] syndrome;
    reg syndrome_valid;
    
    // Iteration counter
    reg [4:0] iter_count;
    
    // Clock gating signals
    reg clk_gate_vn, clk_gate_cn, clk_gate_syndrome;
    wire clk_vn, clk_cn, clk_syndrome;
    
    // Power-aware clock gating
    clock_gate cg_vn (.clk(clk), .enable(clk_gate_vn & ~low_power_mode), .clk_out(clk_vn));
    clock_gate cg_cn (.clk(clk), .enable(clk_gate_cn & ~low_power_mode), .clk_out(clk_cn));
    clock_gate cg_syndrome (.clk(clk), .enable(clk_gate_syndrome), .clk_out(clk_syndrome));
    
    // Early termination logic
    reg early_term_flag;
    reg [3:0] zero_syndrome_count;
    
    // Variable node update module
    variable_node_processor #(
        .NODE_COUNT(BLOCK_SIZE),
        .LLR_WIDTH(LLR_WIDTH),
        .DEGREE(3)
    ) vn_processor (
        .clk(clk_vn),
        .rst_n(rst_n),
        .enable(state == DECODE),
        .llr_channel(variable_nodes),
        .messages_in(variable_to_check),
        .messages_out(check_to_variable),
        .power_level(power_level),
        .update_done()
    );
    
    // Check node update module (Min-Sum)
    check_node_processor #(
        .NODE_COUNT(CHECK_NODES),
        .LLR_WIDTH(LLR_WIDTH),
        .DEGREE(6),
        .MIN_SUM_OFFSET(MIN_SUM_OFFSET)
    ) cn_processor (
        .clk(clk_cn),
        .rst_n(rst_n),
        .enable(state == DECODE),
        .messages_in(check_to_variable),
        .messages_out(variable_to_check),
        .syndrome(syndrome),
        .power_level(power_level)
    );
    
    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // Next state logic
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (llr_valid && enable)
                    next_state = LOAD;
            end
            
            LOAD: begin
                next_state = DECODE;
            end
            
            DECODE: begin
                if (iter_count >= MAX_ITER || early_term_flag)
                    next_state = CHECK;
            end
            
            CHECK: begin
                if (syndrome_valid)
                    next_state = OUTPUT;
            end
            
            OUTPUT: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Main decoder logic with power optimizations
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            iter_count <= 0;
            decode_done <= 0;
            decode_valid <= 0;
            decoder_idle <= 1;
            early_term_flag <= 0;
            zero_syndrome_count <= 0;
            clk_gate_vn <= 0;
            clk_gate_cn <= 0;
            clk_gate_syndrome <= 0;
        end else begin
            case (state)
                IDLE: begin
                    decoder_idle <= 1;
                    decode_done <= 0;
                    decode_valid <= 0;
                    iter_count <= 0;
                    early_term_flag <= 0;
                    zero_syndrome_count <= 0;
                    
                    // Power down all units
                    clk_gate_vn <= 0;
                    clk_gate_cn <= 0;
                    clk_gate_syndrome <= 0;
                end
                
                LOAD: begin
                    decoder_idle <= 0;
                    
                    // Load input LLRs with reduced precision in low power mode
                    if (low_power_mode && power_level > 3) begin
                        // Truncate LSBs for lower power
                        for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
                            variable_nodes[i] <= llr_in[i] & ~((1 << power_level[1:0]) - 1);
                        end
                    end else begin
                        for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
                            variable_nodes[i] <= llr_in[i];
                        end
                    end
                    
                    // Enable processing units
                    clk_gate_vn <= 1;
                    clk_gate_cn <= 1;
                end
                
                DECODE: begin
                    iter_count <= iter_count + 1;
                    
                    // Dynamic iteration reduction in low power mode
                    if (low_power_mode) begin
                        // Check for early termination more aggressively
                        if (|syndrome == 0) begin
                            zero_syndrome_count <= zero_syndrome_count + 1;
                            if (zero_syndrome_count >= (EARLY_TERM_THRESH >> power_level[2:1])) begin
                                early_term_flag <= 1;
                            end
                        end else begin
                            zero_syndrome_count <= 0;
                        end
                    end else begin
                        // Normal early termination
                        if (|syndrome == 0 && iter_count > 3) begin
                            early_term_flag <= 1;
                        end
                    end
                    
                    // Adaptive clock gating based on convergence
                    if (iter_count > 10 && low_power_mode) begin
                        // Reduce activity in later iterations
                        clk_gate_vn <= ~clk_gate_vn;  // Toggle clock
                        clk_gate_cn <= ~clk_gate_cn;
                    end
                end
                
                CHECK: begin
                    clk_gate_syndrome <= 1;
                    syndrome_valid <= 1;
                    iterations_used <= iter_count;
                    
                    // Final hard decision
                    for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
                        decoded_bits[i] <= (variable_nodes[i] < 0) ? 1'b1 : 1'b0;
                    end
                end
                
                OUTPUT: begin
                    decode_done <= 1;
                    decode_valid <= 1;
                    
                    // Power down units
                    clk_gate_vn <= 0;
                    clk_gate_cn <= 0;
                    clk_gate_syndrome <= 0;
                end
            endcase
        end
    end
    
endmodule

// Variable Node Processor with power optimization
module variable_node_processor #(
    parameter NODE_COUNT = 672,
    parameter LLR_WIDTH = 6,
    parameter DEGREE = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire signed [LLR_WIDTH-1:0] llr_channel [0:NODE_COUNT-1],
    input wire signed [LLR_WIDTH-1:0] messages_in [0:NODE_COUNT-1][0:DEGREE-1],
    output reg signed [LLR_WIDTH-1:0] messages_out [0:NODE_COUNT-1][0:DEGREE-1],
    input wire [2:0] power_level,
    output reg update_done
);
    
    reg [9:0] node_counter;
    reg signed [LLR_WIDTH+2:0] sum_total;
    reg processing;
    
    // Reduced precision adder tree for low power
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            node_counter <= 0;
            update_done <= 0;
            processing <= 0;
        end else if (enable) begin
            if (!processing) begin
                processing <= 1;
                node_counter <= 0;
            end else if (node_counter < NODE_COUNT) begin
                // Variable node update with power-aware precision
                sum_total = llr_channel[node_counter];
                
                // Add all incoming messages
                for (integer d = 0; d < DEGREE; d = d + 1) begin
                    if (power_level < 4) begin
                        sum_total = sum_total + messages_in[node_counter][d];
                    end else begin
                        // Reduced precision addition for low power
                        sum_total = sum_total + (messages_in[node_counter][d] >>> power_level[1:0]);
                    end
                end
                
                // Generate outgoing messages
                for (integer d = 0; d < DEGREE; d = d + 1) begin
                    messages_out[node_counter][d] <= saturate(sum_total - messages_in[node_counter][d]);
                end
                
                node_counter <= node_counter + 1;
            end else begin
                update_done <= 1;
                processing <= 0;
            end
        end
    end
    
    // Saturation function
    function signed [LLR_WIDTH-1:0] saturate;
        input signed [LLR_WIDTH+2:0] value;
        begin
            if (value > (2**(LLR_WIDTH-1) - 1))
                saturate = 2**(LLR_WIDTH-1) - 1;
            else if (value < -(2**(LLR_WIDTH-1)))
                saturate = -(2**(LLR_WIDTH-1));
            else
                saturate = value[LLR_WIDTH-1:0];
        end
    endfunction
    
endmodule

// Check Node Processor with Min-Sum algorithm
module check_node_processor #(
    parameter NODE_COUNT = 336,
    parameter LLR_WIDTH = 6,
    parameter DEGREE = 6,
    parameter MIN_SUM_OFFSET = 1
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire signed [LLR_WIDTH-1:0] messages_in [0:NODE_COUNT-1][0:DEGREE-1],
    output reg signed [LLR_WIDTH-1:0] messages_out [0:NODE_COUNT-1][0:DEGREE-1],
    output reg [NODE_COUNT-1:0] syndrome,
    input wire [2:0] power_level
);
    
    reg [8:0] node_counter;
    reg signed [LLR_WIDTH-1:0] min1, min2;
    reg [2:0] min1_idx;
    reg sign_product;
    
    // Power-optimized min-sum processing
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            node_counter <= 0;
            syndrome <= 0;
        end else if (enable && node_counter < NODE_COUNT) begin
            // Find two minimum values with reduced comparisons in low power mode
            min1 = 6'h1F;  // Max positive value
            min2 = 6'h1F;
            min1_idx = 0;
            sign_product = 0;
            
            // Power-aware processing: skip some comparisons at high power levels
            for (integer d = 0; d < DEGREE; d = d + 1) begin
                if (power_level < 5 || (d & ((1 << power_level[1:0]) - 1)) == 0) begin
                    reg signed [LLR_WIDTH-1:0] abs_val;
                    abs_val = (messages_in[node_counter][d] < 0) ? 
                              -messages_in[node_counter][d] : messages_in[node_counter][d];
                    
                    if (abs_val < min1) begin
                        min2 = min1;
                        min1 = abs_val;
                        min1_idx = d;
                    end else if (abs_val < min2) begin
                        min2 = abs_val;
                    end
                    
                    sign_product = sign_product ^ messages_in[node_counter][d][LLR_WIDTH-1];
                end
            end
            
            // Apply offset correction
            min1 = (min1 > MIN_SUM_OFFSET) ? min1 - MIN_SUM_OFFSET : 0;
            min2 = (min2 > MIN_SUM_OFFSET) ? min2 - MIN_SUM_OFFSET : 0;
            
            // Generate output messages
            for (integer d = 0; d < DEGREE; d = d + 1) begin
                reg sign_out;
                reg signed [LLR_WIDTH-1:0] mag_out;
                
                sign_out = sign_product ^ messages_in[node_counter][d][LLR_WIDTH-1];
                mag_out = (d == min1_idx) ? min2 : min1;
                
                messages_out[node_counter][d] <= sign_out ? -mag_out : mag_out;
            end
            
            // Update syndrome
            syndrome[node_counter] <= sign_product;
            
            node_counter <= node_counter + 1;
        end
    end
    
endmodule

// Clock gating cell for power reduction
module clock_gate (
    input wire clk,
    input wire enable,
    output wire clk_out
);
    
    reg enable_latch;
    
    // Latch enable signal on negative edge
    always @(clk or enable) begin
        if (!clk)
            enable_latch <= enable;
    end
    
    // Gate clock
    assign clk_out = clk & enable_latch;
    
endmodule

// Testbench for LDPC decoder
module ldpc_decoder_tb;
    
    parameter BLOCK_SIZE = 672;
    parameter CHECK_NODES = 336;
    parameter LLR_WIDTH = 6;
    
    reg clk, rst_n, enable;
    reg signed [LLR_WIDTH-1:0] llr_in [0:BLOCK_SIZE-1];
    reg llr_valid;
    reg low_power_mode;
    reg [2:0] power_level;
    
    wire [BLOCK_SIZE-1:0] decoded_bits;
    wire decode_done, decode_valid, decoder_idle;
    wire [4:0] iterations_used;
    
    // DUT instantiation
    ldpc_decoder_low_power #(
        .BLOCK_SIZE(BLOCK_SIZE),
        .CHECK_NODES(CHECK_NODES),
        .LLR_WIDTH(LLR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .llr_in(llr_in),
        .llr_valid(llr_valid),
        .decoded_bits(decoded_bits),
        .decode_done(decode_done),
        .decode_valid(decode_valid),
        .iterations_used(iterations_used),
        .low_power_mode(low_power_mode),
        .power_level(power_level),
        .decoder_idle(decoder_idle)
    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Test stimulus
    initial begin
        rst_n = 0;
        enable = 0;
        llr_valid = 0;
        low_power_mode = 0;
        power_level = 0;
        
        // Initialize LLRs with test pattern
        for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
            llr_in[i] = $random % 32 - 16;  // Random LLRs
        end
        
        // Reset
        #20 rst_n = 1;
        #10;
        
        // Test 1: Normal mode decoding
        $display("Test 1: Normal mode decoding");
        enable = 1;
        llr_valid = 1;
        #10 llr_valid = 0;
        
        wait(decode_done);
        $display("Decoding complete. Iterations: %d", iterations_used);
        #20;
        
        // Test 2: Low power mode
        $display("Test 2: Low power mode");
        low_power_mode = 1;
        power_level = 3;
        
        // New codeword
        for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
            llr_in[i] = $random % 32 - 16;
        end
        
        llr_valid = 1;
        #10 llr_valid = 0;
        
        wait(decode_done);
        $display("Low power decoding complete. Iterations: %d", iterations_used);
        
        // Test 3: Ultra-low power mode
        $display("Test 3: Ultra-low power mode");
        power_level = 7;
        
        // Strong codeword for better convergence
        for (integer i = 0; i < BLOCK_SIZE; i = i + 1) begin
            llr_in[i] = (i % 2) ? 15 : -15;  // Strong LLRs
        end
        
        llr_valid = 1;
        #10 llr_valid = 0;
        
        wait(decode_done);
        $display("Ultra-low power decoding complete. Iterations: %d", iterations_used);
        
        #100 $finish;
    end
    
    // Monitor power state
    always @(posedge clk) begin
        if (decoder_idle)
            $display("Time %t: Decoder idle (low power)", $time);
    end
    
endmodule