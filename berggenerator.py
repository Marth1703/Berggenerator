import bpy
import mathutils
import random
from bpy_extras.io_utils import ImportHelper
import numpy as np

bpy.ops.mesh.primitive_plane_add(size = 1)

bpy.ops.object.mode_set(mode='OBJECT')
for o in bpy.context.scene.objects:
   o.hide_set(False)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)
bpy.ops.outliner.orphans_purge()

mountainSize = 2


def createMountain (path, tree_path, snow,grass, scale, treeAmount, treeScale, subdivision, riverLength, createRiver, useCV2, displacementStrength,maximumTreeHeight,minimumRiverHeight, trench_depth, closeKernelSize, openKernelSize):
    
    heightmapPath = path
    sceneryTree1Path = tree_path
    rockNormalPath = "..//01rock_normal.jpg"
    rockRoughnessPath = "..//02rock_roughness.jpg"
    rockBaseColorPath = "..//03rock_basecolor.jpg"
    rockambOcclusionPath = "..//04rock_ambientOcclusion.jpg"
    snowNormalPath = "..//05snow_normal.jpg"
    snowRoughnessPath = "..//06snow_roughness.jpg"
    snowBaseColorPath = "..//07snow_basecolor.jpg"
    snowambOcclusionPath = "..//08snow_ambientOcclusion.jpg" 
    grassNormalPath = "..//09grass_normal.jpg"
    grassRoughnessPath = "..//10grass_roughness.jpg"
    grassBaseColorPath = "..//11grass_basecolor.jpg"
    grassambOcclusionPath = "..//12grass_ambientOcclusion.jpg"

    baseContext = bpy.context

    baseContext.scene.render.engine = 'CYCLES'
    baseContext.scene.cycles.device = 'GPU'

    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    bpy.ops.mesh.primitive_plane_add(size = mountainSize, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    mountainPlaneObj = baseContext.active_object
    mountainPlaneObj.matrix_local.translation = (0,0,1)
    mountainPlaneObj.scale = (int(scale), int(scale), int(scale))

    heightTexture = bpy.data.textures.new('textureMountain', type ='IMAGE')
    heightTextureImage = bpy.data.images.load(filepath=heightmapPath)
    heightTexture.image = heightTextureImage
    
    heightTexture.image.colorspace_settings.name = 'Linear'

    displacementModif = mountainPlaneObj.modifiers.new("Displace", type='DISPLACE')
    displacementModif.texture = heightTexture
    displacementModif.strength = displacementStrength

    bpy.ops.object.shade_smooth()

    bpy.ops.object.editmode_toggle()

    bpy.ops.mesh.subdivide(number_cuts= int(subdivision))
    bpy.ops.mesh.subdivide(number_cuts=10)

    if createRiver:
        textureWithRiver = bpy.data.textures.new('riverTex', type ='IMAGE')
        rHeight, rWidth = heightTextureImage.size
        riverImg = bpy.data.images.new("heightMapRiver", width=rWidth, height=rHeight)
        riverMap = np.array(heightTextureImage.pixels)
        riverMap = np.reshape(riverMap, (rHeight, rWidth, 4))      
       
        max_moves = riverLength
        riverVisits = np.zeros((riverMap.shape[0], riverMap.shape[1]), dtype=bool)
           
        def is_valid(x, y, height, width):
            return x >= 0 and y >= 0 and x < height and y < width
           
        def is_visited(x, y, visited):
            return visited[x][y]

        def visit_pixel(x, y, heightmap, visited, trench_depth):
            visited[x][y] = True
            heightmap[x,y, 0] -= trench_depth
            heightmap[x,y, 1] -= trench_depth
            heightmap[x,y, 2] -= trench_depth
            riverVisits[x][y] = True
           
        def traverse_pixels(heightmap, trench_depth, max_moves):
            moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            queue = []
            start_x = None
            start_y = None
         
            possible_starts = [(x, y) for x in range(heightmap.shape[0]) for y in range(heightmap.shape[1]) if heightmap[x, y, 0] > minimumRiverHeight]
            if len(possible_starts) > 0:
                start_x, start_y = possible_starts[np.random.randint(0, len(possible_starts))]
            else:    
                start_x = random.randint(0, heightmap.shape[0])
                start_y = random.randint(0, heightmap.shape[1])
            visited = np.zeros((heightmap.shape[0], heightmap.shape[1]), dtype=bool)
            visit_pixel(start_x, start_y, heightmap, visited, trench_depth)
            queue.append((start_x, start_y))
            moves_made = 0
            while len(queue) > 0 and moves_made < max_moves:
                x, y = queue.pop(0)
                for move in moves:
                    next_x = x + move[0]
                    next_y = y + move[1]
                    if is_valid(next_x, next_y, heightmap.shape[0], heightmap.shape[1]) and not is_visited(next_x, next_y, visited):
                        next_height = heightmap[next_x, next_y, 0]
                        next_index = None
                        for i in range(len(queue)):
                            if heightmap[queue[i][0], queue[i][1], 0] > next_height:
                                next_index = i
                                break
                        if next_index:
                            queue.insert(next_index, (next_x, next_y))
                        else:
                            queue.append((next_x, next_y))
                        prob_move = 1 - abs(heightmap[x, y, 0] - heightmap[next_x, next_y, 0])
                        if np.random.uniform(0, 1) < prob_move:
                            visit_pixel(next_x, next_y, riverMap, visited, trench_depth)
                            queue.append((next_x, next_y))
                            moves_made += 1

        traverse_pixels(riverMap, trench_depth, max_moves)
            
        riverImg.pixels = riverMap.ravel()
        textureWithRiver.image = riverImg

        displacementModif.texture = textureWithRiver

    else:
        riverVisits = np.zeros((heightTextureImage.size[0], heightTextureImage.size[1]), dtype=bool)


    if useCV2:
        import cv2
        heightImg = cv2.imread(heightmapPath, cv2.IMREAD_GRAYSCALE) 
        
        grayValues = np.array(heightImg)
        grayValues = np.rot90(grayValues, -1)
        
        laplacian = cv2.Laplacian(heightImg, cv2.CV_64F)
        closeKernelSize= int(closeKernelSize)
        openKernelSize=int(openKernelSize)
        closeKernel = np.ones((closeKernelSize, closeKernelSize),np.uint8)
        openKernel = np.ones((openKernelSize, openKernelSize),np.uint8)
        closing = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, closeKernel)
        open = cv2.morphologyEx(closing, cv2.MORPH_OPEN, openKernel)
        img_binary = cv2.threshold(open, 0, 1, cv2.THRESH_BINARY)[1]
        img_binary = np.rot90(img_binary, -1)

        imageArr = np.array(img_binary)
        gradHeight = imageArr.shape[0]
        gradWidth = imageArr.shape[1]
        neededSize = gradHeight
        
        if gradHeight != gradWidth:
            neededSize = gradWidth
            if gradHeight < gradWidth:
                neededSize = gradHeight
            img_binary = cv2.resize(img_binary, (neededSize, neededSize), interpolation = cv2.INTER_LINEAR)


        vertex_group = mountainPlaneObj.vertex_groups.new(name="Gradient")

        verts = []    

        bpy.ops.object.mode_set(mode = 'OBJECT')

        for ver in mountainPlaneObj.data.vertices:
            x = ver.co.x
            y = ver.co.y
            img_x = int((x + 1) * (neededSize / 2))
            img_y = int((y + 1) * (neededSize / 2))
            if 0 <= img_x < neededSize and 0 <= img_y < neededSize:
                if img_binary[img_x, img_y] < 1.0:
                    if grayValues[img_x, img_y] < int(maximumTreeHeight) and not riverVisits[img_y][img_x]:
                        verts.append(ver.index)
        vertex_group.add(verts, 1.0, 'ADD')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    def add_Particle_Scenery(number, amount, scale):
        
        bpy.ops.import_scene.obj(filepath=sceneryTree1Path)
        tree_name = bpy.context.selected_objects[0].name
        bpy.ops.object.select_all(action='DESELECT')
        Tree1 = bpy.data.objects[tree_name]
        particleSettingsName = tree_name + "_particles"
        mountainPlaneObj.modifiers.new(particleSettingsName, type='PARTICLE_SYSTEM')
        ps = mountainPlaneObj.particle_systems[number].settings
        ps.type ='HAIR'
        ps.use_advanced_hair = True
        ps.render_type = 'OBJECT'
        ps.instance_object = Tree1
        ps.count = amount
        if useCV2:
            mountainPlaneObj.particle_systems[number].vertex_group_density = "Gradient"
        mountainPlaneObj.particle_systems[particleSettingsName].seed = random.randint(0, 100)
        Tree1.hide_set(state=True)
        Tree1.scale = (scale, scale, scale)

    add_Particle_Scenery(0, int(treeAmount), treeScale)

    bpy.ops.object.mode_set(mode = 'EDIT')

    mat = bpy.data.materials.new(name="mountainTexture")
    mat.use_nodes = True
    mat.cycles.displacement_method = 'DISPLACEMENT'
    allNodes = mat.node_tree.nodes

    principledRock = allNodes.new(type="ShaderNodeBsdfPrincipled")
    principledSnow = allNodes.new(type="ShaderNodeBsdfPrincipled")
    principledGrass = allNodes.new(type="ShaderNodeBsdfPrincipled")
    matRockSnowOutput = allNodes.new(type="ShaderNodeOutputMaterial")
    matRockGrassOutput = allNodes.new(type="ShaderNodeOutputMaterial")
    matAllOutput = allNodes.new(type="ShaderNodeOutputMaterial")
    materialRock = allNodes.new(type="ShaderNodeOutputMaterial")
    materialSnow = allNodes.new(type="ShaderNodeOutputMaterial")
    materialGrass = allNodes.new(type="ShaderNodeOutputMaterial")
    textImage = allNodes.new(type="ShaderNodeTexImage")
    rockNormal = allNodes.new(type="ShaderNodeTexImage")
    rockRoughness = allNodes.new(type="ShaderNodeTexImage")
    rockBaseColor = allNodes.new(type="ShaderNodeTexImage")
    rockambOcclusion = allNodes.new(type="ShaderNodeTexImage")
    snowNormal = allNodes.new(type="ShaderNodeTexImage")
    snowRoughness = allNodes.new(type="ShaderNodeTexImage")
    snowBaseColor = allNodes.new(type="ShaderNodeTexImage")
    snowambOcclusion = allNodes.new(type="ShaderNodeTexImage")
    grassNormal = allNodes.new(type="ShaderNodeTexImage")
    grassRoughness = allNodes.new(type="ShaderNodeTexImage")
    grassBaseColor = allNodes.new(type="ShaderNodeTexImage")
    grassambOcclusion = allNodes.new(type="ShaderNodeTexImage")
    cRampSnow = allNodes.new(type="ShaderNodeValToRGB")
    cRampGrass = allNodes.new(type="ShaderNodeValToRGB")
    bump = allNodes.new(type="ShaderNodeBump")
    separateColor = allNodes.new(type="ShaderNodeSeparateColor")
    mixShaderRockSnow = allNodes.new(type="ShaderNodeMixShader")
    mixShaderRockGrass = allNodes.new(type="ShaderNodeMixShader")
    mixShaderAll = allNodes.new(type="ShaderNodeMixShader")
    texCoordinateRock = allNodes.new(type="ShaderNodeTexCoord")
    texCoordinateSnow = allNodes.new(type="ShaderNodeTexCoord")
    texCoordinateGrass = allNodes.new(type="ShaderNodeTexCoord")
    mappingRock = allNodes.new(type="ShaderNodeMapping")
    mappingSnow = allNodes.new(type="ShaderNodeMapping")
    mappingGrass = allNodes.new(type="ShaderNodeMapping")
    normalMapRock = allNodes.new(type="ShaderNodeNormalMap")
    normalMapSnow = allNodes.new(type="ShaderNodeNormalMap")
    normalMapGrass = allNodes.new(type="ShaderNodeNormalMap")
    mixColorRock = allNodes.new(type="ShaderNodeMix")
    mixColorSnow = allNodes.new(type="ShaderNodeMix")
    mixColorGrass = allNodes.new(type="ShaderNodeMix")

    rockNormal.image = bpy.data.images.load(filepath=rockNormalPath)
    rockNormal.image.colorspace_settings.name = 'Non-Color'
    rockRoughness.image = bpy.data.images.load(filepath=rockRoughnessPath)
    rockRoughness.image.colorspace_settings.name = 'Non-Color'
    rockBaseColor.image = bpy.data.images.load(filepath=rockBaseColorPath)
    rockBaseColor.image.colorspace_settings.name = 'sRGB'
    rockambOcclusion.image = bpy.data.images.load(filepath=rockambOcclusionPath)
    rockambOcclusion.image.colorspace_settings.name = 'Non-Color'

    snowNormal.image = bpy.data.images.load(filepath=snowNormalPath)
    snowNormal.image.colorspace_settings.name = 'Non-Color'
    snowRoughness.image = bpy.data.images.load(filepath=snowRoughnessPath)
    snowRoughness.image.colorspace_settings.name = 'Non-Color'
    snowBaseColor.image = bpy.data.images.load(filepath=snowBaseColorPath)
    snowBaseColor.image.colorspace_settings.name = 'sRGB'
    snowambOcclusion.image = bpy.data.images.load(filepath=snowambOcclusionPath)
    snowambOcclusion.image.colorspace_settings.name = 'Non-Color'

    grassNormal.image = bpy.data.images.load(filepath=grassNormalPath)
    grassNormal.image.colorspace_settings.name = 'Non-Color'
    grassRoughness.image = bpy.data.images.load(filepath=grassRoughnessPath)
    grassRoughness.image.colorspace_settings.name = 'Non-Color'
    grassBaseColor.image = bpy.data.images.load(filepath=grassBaseColorPath)
    grassBaseColor.image.colorspace_settings.name = 'sRGB'
    grassambOcclusion.image = bpy.data.images.load(filepath=grassambOcclusionPath)
    grassambOcclusion.image.colorspace_settings.name = 'Non-Color'

    textImage.image = bpy.data.images.load(filepath=heightmapPath)
    textImage.image.colorspace_settings.name = 'Linear'

    mixColorRock.data_type = 'RGBA'
    mixColorRock.blend_type = 'ADD'
    mixColorRock.inputs[6].default_value = (0.049, 0.049, 0.049, 1)

    mixColorSnow.data_type = 'RGBA'
    mixColorSnow.blend_type = 'ADD'
    mixColorSnow.inputs[6].default_value = (0.367, 0.367, 0.367, 1)

    mixColorGrass.data_type = 'RGBA'
    mixColorGrass.blend_type = 'ADD'
    mixColorGrass.inputs[6].default_value = (0.021, 0.121, 0.022, 1)


    #Mountain Texture
    mat.node_tree.links.new(texCoordinateRock.outputs[0], mappingRock.inputs[0])
    mat.node_tree.links.new(mappingRock.outputs[0], rockNormal.inputs[0])
    mat.node_tree.links.new(rockNormal.outputs[0], normalMapRock.inputs[1])
    mat.node_tree.links.new(normalMapRock.outputs[0], principledRock.inputs[22])
    mat.node_tree.links.new(mappingRock.outputs[0], rockRoughness.inputs[0])
    mat.node_tree.links.new(rockRoughness.outputs[0], principledRock.inputs[9])
    mat.node_tree.links.new(principledRock.outputs[0], materialRock.inputs[0])
    mat.node_tree.links.new(mappingRock.outputs[0], rockBaseColor.inputs[0])
    mat.node_tree.links.new(mappingRock.outputs[0], rockambOcclusion.inputs[0])
    mat.node_tree.links.new(rockBaseColor.outputs[0], mixColorRock.inputs[7])
    mat.node_tree.links.new(rockambOcclusion.outputs[0], mixColorRock.inputs[0])
    mat.node_tree.links.new(mixColorRock.outputs[2], principledRock.inputs[0])
    mat.node_tree.links.new(principledRock.outputs[0], mixShaderRockGrass.inputs[1])
    mat.node_tree.links.new(principledRock.outputs[0], mixShaderRockSnow.inputs[1])

    mat.node_tree.links.new(texCoordinateSnow.outputs[0], mappingSnow.inputs[0])
    mat.node_tree.links.new(mappingSnow.outputs[0], snowNormal.inputs[0])
    mat.node_tree.links.new(snowNormal.outputs[0], normalMapSnow.inputs[1])
    mat.node_tree.links.new(normalMapSnow.outputs[0], principledSnow.inputs[22])
    mat.node_tree.links.new(mappingSnow.outputs[0], snowRoughness.inputs[0])
    mat.node_tree.links.new(snowRoughness.outputs[0], principledSnow.inputs[9])
    mat.node_tree.links.new(principledSnow.outputs[0], materialSnow.inputs[0])
    mat.node_tree.links.new(mappingSnow.outputs[0], snowBaseColor.inputs[0])
    mat.node_tree.links.new(mappingSnow.outputs[0], snowambOcclusion.inputs[0])
    mat.node_tree.links.new(snowBaseColor.outputs[0], mixColorSnow.inputs[7])
    mat.node_tree.links.new(snowambOcclusion.outputs[0], mixColorSnow.inputs[0])
    mat.node_tree.links.new(mixColorSnow.outputs[2], principledSnow.inputs[0])
    mat.node_tree.links.new(principledSnow.outputs[0], mixShaderRockSnow.inputs[2])

    mat.node_tree.links.new(texCoordinateGrass.outputs[0], mappingGrass.inputs[0])
    mat.node_tree.links.new(mappingGrass.outputs[0], grassNormal.inputs[0])
    mat.node_tree.links.new(grassNormal.outputs[0], normalMapGrass.inputs[1])
    mat.node_tree.links.new(normalMapGrass.outputs[0], principledGrass.inputs[22])
    mat.node_tree.links.new(mappingGrass.outputs[0], grassRoughness.inputs[0])
    mat.node_tree.links.new(grassRoughness.outputs[0], principledGrass.inputs[9])
    mat.node_tree.links.new(principledGrass.outputs[0], materialGrass.inputs[0])
    mat.node_tree.links.new(mappingGrass.outputs[0], grassBaseColor.inputs[0])
    mat.node_tree.links.new(mappingGrass.outputs[0], grassambOcclusion.inputs[0])
    mat.node_tree.links.new(grassBaseColor.outputs[0], mixColorGrass.inputs[7])
    mat.node_tree.links.new(grassambOcclusion.outputs[0], mixColorGrass.inputs[0])
    mat.node_tree.links.new(mixColorGrass.outputs[2], principledGrass.inputs[0])
    mat.node_tree.links.new(principledGrass.outputs[0], mixShaderRockGrass.inputs[2])

    mat.node_tree.links.new(separateColor.outputs[2], cRampSnow.inputs[0])
    mat.node_tree.links.new(separateColor.outputs[2], cRampGrass.inputs[0])
    mat.node_tree.links.new(bump.outputs[0], separateColor.inputs[0])
    mat.node_tree.links.new(cRampGrass.outputs[0], mixShaderRockGrass.inputs[0])
    mat.node_tree.links.new(cRampSnow.outputs[0], mixShaderRockSnow.inputs[0])
    mat.node_tree.links.new(mixShaderRockGrass.outputs[0], matRockGrassOutput.inputs[0])
    mat.node_tree.links.new(mixShaderRockSnow.outputs[0], matRockSnowOutput.inputs[0])
    mat.node_tree.links.new(mixShaderRockSnow.outputs[0], mixShaderAll.inputs[2])
    mat.node_tree.links.new(mixShaderRockGrass.outputs[0], mixShaderAll.inputs[1])
    mat.node_tree.links.new(mixShaderAll.outputs[0], matAllOutput.inputs[0])

    bpy.context.active_object.data.materials.append(mat)

    deleteMatOutput =  mat.node_tree.nodes['Material Output']
    mat.node_tree.nodes.remove(deleteMatOutput)
    deleteBSDF =  mat.node_tree.nodes['Principled BSDF']
    mat.node_tree.nodes.remove(deleteBSDF)

    cRampGrass.color_ramp.elements[0].position = 1-grass
    cRampGrass.color_ramp.elements[0].color = (0.077,0.078,0.079,1)
    cRampGrass.color_ramp.elements[1].position = (0.99)

    cRampSnow.color_ramp.elements[0].position = snow -0.01
    cRampSnow.color_ramp.elements[1].color = (0.077,0.078,0.079,1)
    cRampSnow.color_ramp.elements[0].color = (1,1,1,1)
    cRampSnow.color_ramp.elements[1].position = (snow)

    mappingRock.inputs[3].default_value[0] = 12
    mappingRock.inputs[3].default_value[1] = 12
    mappingRock.inputs[3].default_value[2] = 12

    mappingSnow.inputs[3].default_value[0] = 3
    mappingSnow.inputs[3].default_value[1] = 3
    mappingSnow.inputs[3].default_value[2] = 3

    mappingGrass.inputs[3].default_value[0] = 12
    mappingGrass.inputs[3].default_value[1] = 12
    mappingGrass.inputs[3].default_value[2] = 12

    materialSnow.location = (400, 2700)
    normalMapSnow.location = (-100, 2000)
    principledSnow.location = (100, 2600)
    mixColorSnow.location = (-100, 2500)
    snowambOcclusion.location = (-400, 2800)
    snowBaseColor.location = (-400, 2500)
    snowRoughness.location = (-400, 2200)
    snowNormal.location = (-400, 1900)
    mappingSnow.location = (-700, 2420)
    texCoordinateSnow.location = (-940, 2400)

    materialRock.location = (400, 1000)
    normalMapRock.location = (-100, 300)
    principledRock.location = (100, 850)
    mixColorRock.location = (-100, 900)
    rockambOcclusion.location = (-400, 1250)
    rockBaseColor.location = (-400, 950)
    rockRoughness.location = (-400, 650)
    rockNormal.location = (-400, 350)
    mappingRock.location = (-700, 570)
    texCoordinateRock.location = (-940, 550)

    grassambOcclusion.location = (-400, -100)
    grassBaseColor.location = (-400, -400)
    grassRoughness.location = (-400, -700)
    grassNormal.location = (-400, -1000)
    mappingGrass.location = (-700, -620)
    texCoordinateGrass.location = (-940, -600)
    principledGrass.location = (100, -300)
    normalMapGrass.location = (-100, -1100)
    mixColorGrass.location = (-100, -200)
    materialGrass.location = (400, -400)

    bump.location = (600, 1250)
    separateColor.location = (850, 1300)
    cRampSnow.location = (1300, 1500)
    cRampGrass.location = (1300, 1000)
    mixShaderRockSnow.location = (1600, 1200)
    mixShaderRockGrass.location = (1600, 700)
    matRockSnowOutput.location = (2000, 1300)
    mixShaderAll.location = (2000, 1100)
    matRockGrassOutput.location = (2000, 800)
    matAllOutput.location = (2300, 1100)
    allNodes.active = matAllOutput

    bpy.ops.object.editmode_toggle()

class ImportImage(bpy.types.Operator, ImportHelper):
    """Import an image file"""
    bl_idname = "import_test.some_data"
    bl_label = "Heightmap auswählen und Berg mit angegebenen Parametern erstellen"

    filename_ext = ".png;.jpg;.jpeg;.tif;.tiff"
    filter_glob = bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff", options={'HIDDEN'})

    def execute(self, context):
        snow = context.scene.snow_amount
        grass = context.scene.grass_amount
        scale =  context.scene.mountain_scale 
        treeAmount = context.scene.tree_amount
        treeScale = context.scene.tree_scale
        subdivision = context.scene.subdivision
        riverLength = context.scene.river_length
        show_river = context.scene.show_river
        use_CV2 = context.scene.use_CV2
        displacementStrength = context.scene.displacementStrength
        maximumTreeHeight = context.scene.maximumTreeHeight
        minimumRiverHeight= context.scene.minimumRiverHeight
        trench_depth = context.scene.trench_depth
        closeKernelSize = context.scene.closeKernelSize
        openKernelSize = context.scene.openKernelSize
        path = self.filepath
        
        createMountain(path, tree_path, snow,grass, scale, treeAmount,treeScale,subdivision , riverLength, show_river,use_CV2, displacementStrength,maximumTreeHeight,minimumRiverHeight,trench_depth,closeKernelSize , openKernelSize )
        
        return {'FINISHED'}

class ImportTreeObj(bpy.types.Operator, ImportHelper):
    """Import an image file"""
    bl_idname = "import_tree.some_data"
    bl_label = "Tree Object auswählen."
    
    
    filename_ext = ".png;.jpg;.jpeg;.tif;.tiff"
    filter_glob = bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff", options={'HIDDEN'})

    def execute(self, context):
        global tree_path
        tree_path = self.filepath
        return {'FINISHED'}

class MountainPanel(bpy.types.Panel):
    bl_label = "Einstellbare Parameter"
    bl_idname = "test"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "snow_amount", slider=True)
        layout.prop(scene, "grass_amount", slider=True)
        layout.prop(scene, "mountain_scale", slider=True)
        layout.prop(scene, "tree_amount", slider=True)
        layout.prop(scene, "tree_scale", slider=True)
        layout.prop(scene, "subdivision", slider=True)
        layout.prop(scene, "river_length", slider=True)
        layout.prop(scene, "displacementStrength", slider=True)
        layout.prop(scene, "show_river", toggle=True)
        layout.prop(scene, "use_CV2", toggle=True)
        layout.prop(scene, "maximumTreeHeight", slider=True)
        layout.prop(scene, "minimumRiverHeight", slider=True)
        layout.prop(scene, "trench_depth", slider=True)
        layout.prop(scene, "closeKernelSize", slider=True)
        layout.prop(scene, "openKernelSize", slider=True)
        layout.operator("import_test.some_data") 
        layout.operator("import_tree.some_data")

bpy.types.Scene.snow_amount = bpy.props.FloatProperty(
    name = "Menge an Schnee",
    description = "Menge an Schnee",
    default = 0,
    min = 0,
    max = 1
)
bpy.types.Scene.grass_amount = bpy.props.FloatProperty(
    name = "Menge an Gras",
    description = "Menge an Gras",
    default = 0.6,
    min = 0,
    max = 0.99
)

bpy.types.Scene.mountain_scale = bpy.props.FloatProperty(
    name = "Größe des Berges",
    description = "Größe des Berges",
    default = 5,
    min = 0,
    max = 200
)
bpy.types.Scene.tree_amount = bpy.props.FloatProperty(
    name = "Anzahl der Bäume",
    description = "Anzahl der Bäume",
    default = 1000,
    min = 0,
    max = 10000
)
bpy.types.Scene.tree_scale = bpy.props.FloatProperty(
    name = "Größe der Bäume",
    description = "Größe der Bäume",
    default = 1,
    min = 0,
    max = 10
)
bpy.types.Scene.subdivision = bpy.props.FloatProperty(
    name = "Anzahl Subdivisions",
    description = "Anzahl Subdivisions",
    default = 10,
    min = 0,
    max = 50
)
bpy.types.Scene.river_length = bpy.props.FloatProperty(
    name = "Länge des Flusses",
    description = "Länge des Flusses",
    default = 20000,
    min = 0,
    max = 2000000
)
bpy.types.Scene.displacementStrength = bpy.props.FloatProperty(
    name = "Stärke des Displacement",
    description = "Stärke des Displacement",
    default = 1,
    min = 0,
    max = 3
)
bpy.types.Scene.maximumTreeHeight = bpy.props.FloatProperty(
    name = "Maximale Baumhöhe",
    description = "Maximale Baumhöhe",
    default = 100,
    min = 0,
    max = 255
)
bpy.types.Scene.minimumRiverHeight = bpy.props.FloatProperty(
    name = "Minimale Starthöhe des Fluss",
    description = "Minimale Starthöhe des Fluss",
    default = 0.4,
    min = 0,
    max = 1
)
bpy.types.Scene.trench_depth = bpy.props.FloatProperty(
    name = "Tiefe des Flusses",
    description = "Tiefe des Flusses",
    default = 0.04,
    min = 0,
    max = 0.5
)
bpy.types.Scene.closeKernelSize = bpy.props.FloatProperty(
    name = "Größe des closing Kernels",
    description = "Größe des closing Kernels",
    default = 4,
    min = 2,
    max = 10
)
bpy.types.Scene.openKernelSize = bpy.props.FloatProperty(
    name = "Größe des opening Kernels",
    description = "Größe des opening Kernels",
    default = 4,
    min = 2,
    max = 10
)
bpy.types.Scene.show_river = bpy.props.BoolProperty(name = "Einen Fluss Erstellen", default=True)
bpy.types.Scene.use_CV2 = bpy.props.BoolProperty(name = "CV2 benutzen für bessere Baumplatzierung ?", default=False)


bpy.utils.register_class(ImportImage)
bpy.utils.register_class(ImportTreeObj)
bpy.utils.register_class(MountainPanel)