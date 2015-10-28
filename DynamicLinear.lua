require 'nn'

DynamicLinear, parent = torch.class('nn.DynamicLinear', 'nn.Module')

function DynamicLinear:__init(inputSize, outputSize)
    self.module = nn.Linear(inputSize, outputSize)
    self.module_params, self.module_gradParams = self.module:getParameters()
end

function DynamicLinear:updateOutput(input)
    local new_params = input[1]
    new_params = new_params:view(new_params:nElement())
    local current_input = input[2]

    self.module_params:copy(new_params)
    self.output = self.module:forward(current_input)
    return self.output
end

function DynamicLinear:updateGradInput(input, gradOutput)
    local new_params = input[1]
    local current_input = input[2]

    local grad_current_input = self.module:backward(current_input, gradOutput)

    self.gradInput = {}
    self.gradInput[1] = torch.Tensor(input[1]:size()):copy(self.module_gradParams)
    self.gradInput[2] = grad_current_input:clone()
    return self.gradInput
end
