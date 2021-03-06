require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size)
    -- nngraph.setDebug(true)

    --------------------- input structure ---------------------
    local inputs = {}
    table.insert(inputs, nn.Identity()():annotate{name="input"})   -- network input
    table.insert(inputs, nn.Identity()():annotate{name="c of t-1"})   -- c at time t-1
    table.insert(inputs, nn.Identity()():annotate{name="h of t-1"})   -- h at time t-1
    local input = inputs[1]
    local prev_c = inputs[2]
    local prev_h = inputs[3]

    --------------------- preactivations ----------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(input):annotate{name="i2h"}   -- input to hidden
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name="h2h"}    -- hidden to hidden
    local preactivations = nn.CAddTable()({i2h, h2h}):annotate{name="preactivations"}        -- i2h + h2h

    ------------------ non-linear transforms ------------------
    -- gates
    local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations):annotate{name="pre_sigmoid"}
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk):annotate{name="all_gates"}

    -- input
    local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations):annotate{name="in_chunk"}
    local in_transform = nn.Tanh()(in_chunk):annotate{name="in_transform"}

    ---------------------- gate narrows -----------------------
    local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)

    --------------------- next cell state ---------------------
    local c_forget = nn.CMulTable()({forget_gate, prev_c})  -- previous cell state contribution
    local c_input = nn.CMulTable()({in_gate, in_transform}) -- input contribution
    local next_c = nn.CAddTable()({
        c_forget,
        c_input
    })

    -------------------- next hidden state --------------------
    local c_transform = nn.Tanh()(next_c)
    local next_h = nn.CMulTable()({out_gate, c_transform})

    --------------------- output structure --------------------
    outputs = {}
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)

    -- packs the graph into a convenient module with standard API (:forward(), :backward())
    return nn.gModule(inputs, outputs)
    -- return inputs, outputs
end

return LSTM
