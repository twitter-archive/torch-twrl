-- Based on tilecoding Version3 from https://webdocs.cs.ualberta.ca/~sutton/tiles/tiles3.html
local function tilecoding(opt)
   -- Set a random seed for consistent hashing
   math.randomseed(65597)
   local numTilings = opt.numTilings
   local memorySize = opt.memorySize
   local sizeVal = opt.sizeVal or 2048
   local maxLongInteger = 2147483647
   local maxLongIntegerBy4 = math.floor(maxLongInteger/4)
   local randomTable = {}
   -- initialize the table of random numbers for UNH hashing
   for i = 1,sizeVal do
      -- the range are matching the original python code
      randomTable[i] = math.random(0, maxLongIntegerBy4 - 1)
   end
   local _qstate = {}
   local _base = {}
   local function hashcoords(coordinates, m)
      local increment = 449
      local res = 0
      for i = 1, #coordinates do
         rtIdx = ((coordinates[i] + (i*increment)) % sizeVal) + 1
         res = res + randomTable[rtIdx]
      end
      -- ensure there is no tile at index 0
      return (res % m) + 1
   end
   local tc = {}
   function tc.tiles(memorySize, numTilings, floats, ints)
      local ints = ints or {}
      local Tiles = {}
      local _qstate = {}
      for i = 1, #floats do
         _qstate[i] = math.floor(floats[i] * numTilings)
      end
      -- for each tiling
      for tiling = 1, numTilings do
         local tilingX2 = tiling * 2
         local coords = {tiling}
         local b = tiling
         -- building hashable float to coords
         for q = 1, #_qstate do
            table.insert(coords, math.floor( (_qstate[q] + b) / numTilings))
            b = b + tilingX2
         end
         -- extend hashable coords with ints
         for i = 1, #ints do
            table.insert(coords, ints[i])
         end
         table.insert(Tiles, hashcoords(coords, memorySize))
      end
      return Tiles
   end
   return tc
end
return tilecoding