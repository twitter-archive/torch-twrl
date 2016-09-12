-- Based on tilecoding Version3 from https://webdocs.cs.ualberta.ca/~sutton/tiles/tiles3.html
local function tilecoding(opt)
   -- Set a random seed for consistent hashing
   math.randomseed(65597)
	local numTilings = opt.numTilings
	local scaleFactor = opt.scaleFactor
	local memorySize = opt.memorySize
   local stateMins = opt.stateMins
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
      return res % m
   end
	local tc = {}
   function tc.feature(s)
      local floats = {}
      assert(#s==#scaleFactor,"Dimension of scaling factor and feature vectors must match!")
      for i=1,#s do
         if stateMins[i] then
            floats[i] = (s[i] + stateMins[i]) / scaleFactor[i]
         else
            floats[i] = s[i] / scaleFactor[i]
         end
      end
      local F = tc.tiles(memorySize, numTilings, floats)
      return F
   end
	function tc.tiles(memorySize, numTilings, floats)
      local coords = {}
      local Tiles = {}
      local _qstate = {}
      local b = 0
      for i = 1, #floats do
         _qstate[i] = math.floor(floats[i] * numTilings)
      end
      for tiling = 1,numTilings do -- for each tiling
         local tilingX2 = tiling*2
         table.insert(coords,tiling)
         b = tiling
         for q = 1,#_qstate do
            table.insert(coords, math.floor((_qstate[q] + b) / numTilings))
            b = b + tilingX2
         end
         table.insert(Tiles, hashcoords(coords, memorySize))
      end
      return Tiles
   end
	return tc
end
return tilecoding