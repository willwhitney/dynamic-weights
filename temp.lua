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

-- input = torch.rand(1, 2)
for i = 1, 10000 do
    input = torch.rand(1, 2)
    target = torch.Tensor{{(input[1][1] + input[1][2])/2, (input[1][1] - input[1][2])/2, }}
    -- print(target)

    -- DL:forward(input_table)

    -- DL:backward(input_table, gradOutput)
    -- print(DL:backward(input_table, gradOutput))

    -- print(concat.output)

    output = net:forward(input)
    -- print("Output:", output)
    print("Loss:", criterion:forward(output, target))

    -- print(criterion:backward(net.output, target))

    -- DL_gradInput = DL:backward(concat.output, criterion:backward(output, target))
    -- print(DL_gradInput)
    -- print(concat:backward(input, DL_gradInput))
    net:backward(input, criterion:backward(net.output, target))
    net:updateParameters(2e-1)
    net:zeroGradParameters()
end
