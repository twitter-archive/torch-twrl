-- Based on tilecoding Version3 from https://webdocs.cs.ualberta.ca/~sutton/tiles/tiles3.html
-- TODO: add index hash table functionality
math.randomseed(1337)
local hash = require 'hash'
local function tilecoding(opt)
   -- Set a random seed for consistent hashing
   local tc = {}
   function tc.tiles(memorySize, numTilings, floats, ints)
      local ints = ints or {}
      local Tiles = {}
      local qfloats = {}
      local coords = {}
      for i = 1, #floats do
         qfloats[i] = math.floor(floats[i] * numTilings)
      end
      -- for each tiling
      for tiling = 1, numTilings do
         local tilingX2 = tiling * 2
         coords = {tiling}
         local b = tiling
         -- building hashable float to coords
         for q = 1, #qfloats do
            table.insert(coords, math.floor( ( qfloats[q] + b ) / numTilings))
            b = b + tilingX2
         end
         -- extend hashable coords with ints
         for i = 1, #ints do
            table.insert(coords, ints[i])
         end
         local feature = hash.hash(torch.Tensor(coords)) % memorySize
         table.insert(Tiles, feature)
      end
      return Tiles
   end
   return tc
end
return tilecoding