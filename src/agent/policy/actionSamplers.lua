local function categorical(actions, opt)
   local opt = opt or {}

   local actionShift = opt.actionShift
   return (torch.multinomial(actions, 1) - actionShift)[1][1]
end

local function normal(actions, opt)
   local opt = opt or {}
   local std = opt.std
   local actionBoundFactor = opt.actionBoundFactor
   local actions = torch.cmul(torch.normal(actions, std), actionBoundFactor)

   actions = actions:size():size() == 2 and actions[1] or actions
   return actions:totable()
end

return {
   categorical = categorical,
   normal = normal
}
