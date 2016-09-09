local function getModel(opt)
	local opt = opt or {}
	local nInputs = opt.nInputs
	local nOutputs = opt.nOutputs
	local nHiddenLayerSize = opt.nHiddenLayerSize
	local outputType = opt.outputType or 'categorical'
	print(nHiddenLayerSize)
	local finalLayer = outputType == 'categorical' and nn.SoftMax()
		or nn.Tanh()

	local model = nn.Sequential()
		:add(nn.Linear(nInputs, nHiddenLayerSize))
		:add(nn.Tanh())
		:add(nn.Linear(nHiddenLayerSize, nOutputs))
		:add(finalLayer)

  print(model)
	local theta, gradTheta = model:getParameters()
	local gradThetaSq = torch.Tensor(gradTheta:size()):zero()

	return {
		model = model,
		theta = theta,
		gradTheta = gradTheta,
		gradThetaSq = gradThetaSq
	}
end

return getModel
