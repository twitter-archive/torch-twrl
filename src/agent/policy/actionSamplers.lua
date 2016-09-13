local function categorical(actions, opt)
   local opt = opt or {}

   -- typically the action space is 0 indexed
   local actionShift = opt.actionShift or 1
   local actions = torch.exp(actions)
   return (torch.multinomial(actions, 1) - actionShift)[1][1]
end

local function normal(actions, opt)
   local opt = opt or {}
   local std = opt.std
   -- actionShift is the multiplier for the Continuous action value
   -- TODO: better variable name ?
   local actionShift = opt.actionShift or 1

   local actions = torch.normal(actions, std) * actionShift
   return actions
end

return {
   categorical = categorical,
   normal = normal
}
