local function getModel(opt)
	local opt = opt or {}
	local nInputs = opt.nInputs
	local nOutputs = opt.nOutputs
	local nHiddenLayerSize = opt.nHiddenLayerSize
	local outputType = opt.outputType or 'categorical'
	local finalLayer = outputType == 'categorical' and nn.SoftMax()
		or nn.Tanh()

	local net = nn.Sequential()
		:add(nn.Linear(nInputs, nHiddenLayerSize))
		:add(nn.Tanh())
		:add(nn.Linear(nHiddenLayerSize, nOutputs))
		:add(finalLayer)

	return {
		net = net 
	}
end

return getModel