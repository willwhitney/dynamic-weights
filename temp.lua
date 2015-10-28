require 'DynamicLinear'


net = nn.Sequential()
concat = nn.ConcatTable()
concat:add(nn.Linear(2, 6))
concat:add(nn.Identity())
net:add(concat)

DL = nn.DynamicLinear(2, 2)
net:add(DL)
net:add(nn.Tanh())

criterion = nn.MSECriterion()


for i = 1, 1000 do
    input = torch.rand(1, 2)
    target = torch.zeros(1, 2)


    -- DL:forward(input_table)

    -- DL:backward(input_table, gradOutput)
    -- print(DL:backward(input_table, gradOutput))

    -- print(concat.output)

    output = net:forward(input)
    -- print("Output:", output)
    print("Loss:", criterion:forward(output, target))

    -- print(criterion:backward(net.output, target))

    DL_gradInput = DL:backward(concat.output, criterion:backward(net.output, target))
    -- print(DL_gradInput)
    -- print(concat:backward(input, DL_gradInput))
    net:backward(input, criterion:backward(net.output, target))
    net:updateParameters(2e-9)
end
