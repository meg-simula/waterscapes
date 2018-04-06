#### import the simple module from the paraview
from paraview.simple import *
import os
#### disable automatic camera reset on 'Show'
root = '/home/eleonora/Repositories/waterscapes/src/MPET/sandbox/brain-simulations/results_brain_transfer_1e-06/'
# directories = os.listdir(root)
directories = ["nu_0.4999theta_1.0_formulationtype_standard_solvertype_direct",\
               "nu_0.4999theta_1.0_formulationtype_total_pressure_solvertype_direct" ]
paraview.simple._DisableFirstRenderCameraReset()
for d in directories:
	try:
		# create a new 'PVD Reader'
		path = root + d + '/pvd/'
		p4pvd = PVDReader(FileName = path + 'p4.pvd')
		arrayname = p4pvd.PointData.GetArray(0).Name
		# get animation scene
		animationScene1 = GetAnimationScene()
		# t = 4.50s
		animationScene1.AnimationTime = 90.0
		# update animation scene based on data timesteps
		animationScene1.UpdateAnimationUsingDataTimeSteps()

		# get active view
		renderView1 = GetActiveViewOrCreate('RenderView')
		# uncomment following to set a specific view size
		renderView1.ViewSize = [1216, 801]

		# show data in view
		p4pvdDisplay = Show(p4pvd, renderView1)
		# get color transfer function/color map for arrayname
		f_623LUT = GetColorTransferFunction(arrayname)
		f_623LUTColorBar = GetScalarBar(f_623LUT, renderView1)
		f_623LUTColorBar.AutoOrient = 1
		f_623LUTColorBar.Orientation = 'Vertical'
		f_623LUTColorBar.Position = [0.8709429280397022, 0.5967540574282147]
		f_623LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
		f_623LUTColorBar.TitleOpacity = 1.0
		f_623LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
		f_623LUTColorBar.LabelOpacity = 1.0
		# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
		f_623LUT.ApplyPreset('Reds', True)
		# invert the transfer function
		f_623LUT.InvertTransferFunction()
		# create a new 'Slice'
		slice1 = Slice(Input=p4pvd)
		slice1.SliceType = 'Plane'
		slice1.Crinkleslice = 0
		slice1.Triangulatetheslice = 1
		slice1.SliceOffsetValues = [0.0]

		# init the 'Plane' selected for 'SliceType'
		slice1.SliceType.Origin = [89.905605, 108.860115, 82.30183000000001]
		slice1.SliceType.Normal = [1.0, 0.0, 0.0]
		slice1.SliceType.Offset = 0.0

		# Properties modified on slice1.SliceType
		slice1.SliceType.Normal = [0.0, 1.0, 0.0]

		# Properties modified on slice1.SliceType
		slice1.SliceType.Normal = [0.0, 1.0, 0.0]

		# show data in view
		slice1Display = Show(slice1, renderView1)

		# create a new 'Slice'
		slice2 = Slice(Input=p4pvd)
		slice2.SliceType = 'Plane'
		slice2.Crinkleslice = 0
		slice2.Triangulatetheslice = 1
		slice2.SliceOffsetValues = [0.0]

		# init the 'Plane' selected for 'SliceType'
		slice2.SliceType.Origin = [89.905605, 108.860115, 82.30183000000001]
		slice2.SliceType.Normal = [1.0, 0.0, 0.0]
		slice2.SliceType.Offset = 0.0

		# Properties modified on slice2.SliceType
		slice2.SliceType.Normal = [0.0, 0.0, 1.0]

		# Properties modified on slice2.SliceType
		slice2.SliceType.Normal = [0.0, 0.0, 1.0]

		# show data in view
		slice2Display = Show(slice2, renderView1)
		# Properties modified on p4pvdDisplay
		p4pvdDisplay.Opacity = 0.1

		# show color bar/color legend
		slice1Display.SetScalarBarVisibility(renderView1, True)

		# reset view to fit data
		renderView1.ResetCamera()

		#changing interaction mode based on data extents
		renderView1.InteractionMode = '3D'

		# show color bar/color legend
		p4pvdDisplay.SetScalarBarVisibility(renderView1, True)

		# update the view to ensure updated data information
		renderView1.Update()

		#### saving camera placements for all active views

		# current camera placement for renderView1
		# get active view
		renderView1 = GetActiveViewOrCreate('RenderView')
		# uncomment following to set a specific view size
		# renderView1.ViewSize = [1612, 801]

		# current camera placement for renderView1
		renderView1.CameraPosition = [300.02495279494485, 309.9940888308699, 156.4837887842105]
		renderView1.CameraFocalPoint = [89.905605, 108.860115, 82.30183000000001]
		renderView1.CameraViewUp = [-0.16429190404960017, -0.12435634343351876, 0.9785416036692574]
		renderView1.CameraParallelScale = 137.636609409269
		#### uncomment the following to render all views
		# RenderAllViews()
		# alternatively, if you want to write images, you can use SaveScreenshot(...).
		# Hide orientation axes
		renderView1.OrientationAxesVisibility = 0
        # Data Range
        datarange = p4pvd.PointData.GetArray(0).GetRange()
		f_623LUT.RescaleTransferFunction(datarange[0], datarange[1])
		# update the view to ensure updated data information
		renderView1.Update()

		SaveScreenshot(path + "p4_snap.png", renderView1)
	    # set active source
		SetActiveSource(None)

		# set active view
		SetActiveView(None)
	except:
		print("something went wrong! directory = ", d)

