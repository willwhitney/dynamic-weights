require 'DynamicLinear'

torch.manualSeed(2)

net = nn.Sequential()
concat = nn.ConcatTable()

weight_predictor = nn.Sequential()
weight_predictor:add(nn.Linear(3, 8))

concat:add(weight_predictor)
concat:add(nn.Identity())
net:add(concat)

dynamic = nn.Sequential()
DL = nn.DynamicLinear(3, 2)
dynamic:add(DL)
dynamic:add(nn.Tanh())
net:add(dynamic)

criterion = nn.MSECriterion()

steps = 100000



-- lr = 1e-2
-- for i = 1, steps do
--     input = torch.rand(1, 3)
--     if input[1][3] < 0.5 then
--         target = torch.Tensor{{(input[1][1] + input[1][2])/2, (input[1][1] - input[1][2])/2, }}
--     else
--         target = torch.Tensor{{(5*input[1][1] + input[1][2])/6, (input[1][1] - 3*input[1][2])/4, }}
--     end
--
--     output = net:forward(input)
--     -- print("Loss:", criterion:forward(output, target))
--
--     net:backward(input, criterion:backward(net.output, target))
--     net:updateParameters(lr)
--     net:zeroGradParameters()
-- end

lr = 3e-2
net = nn.Sequential()
net:add(nn.Linear(3,8))
net:add(nn.Tanh())
net:add(nn.Linear(8,2))
net:add(nn.Tanh())

for i = 1, steps do
    input = torch.rand(1, 3)
    if input[1][3] < 0.5 then
        target = torch.Tensor{{(input[1][1] + input[1][2])/2, (input[1][1] - input[1][2])/2, }}
    else
        target = torch.Tensor{{(5*input[1][1] + input[1][2])/6, (input[1][1] - 3*input[1][2])/4, }}
    end

    output = net:forward(input)
    -- print("Loss:", criterion:forward(output, target))

    net:backward(input, criterion:backward(net.output, target))
    net:updateParameters(lr)
    net:zeroGradParameters()
end


loss = 0
for i = 1, 1000 do
    input = torch.rand(1, 3)
    if input[1][3] < 0.5 then
        target = torch.Tensor{{(input[1][1] + input[1][2])/2, (input[1][1] - input[1][2])/2, }}
    else
        target = torch.Tensor{{(5*input[1][1] + input[1][2])/6, (input[1][1] - 3*input[1][2])/4, }}
    end
    output = net:forward(input)

    loss = loss + criterion:forward(output, target)
end
loss = loss / 1000

print(net)
print("final loss:", loss)
