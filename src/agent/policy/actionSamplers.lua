local function categorical(actions, opt)
   local opt = opt or {}
   -- typically the action space is 0 indexed
   local actionShift = opt.actionShift or 1
   return (torch.multinomial(actions, 1) - actionShift)[1][1]
end

local function normal(actions, opt)
   local opt = opt or {}
   local std = opt.std
   -- actionShift is the multiplier for the Continuous action value
   -- TODO: better variable name ?
   local actionShift = opt.actionShift or 1
   local actions = torch.normal(actions, std) * actionShift

   -- TODO: this assumes that this function is not called in batches
   actions = actions:size():size() == 2 and actions[1] or actions
   return actions:totable()
end

return {
   categorical = categorical,
   normal = normal
}