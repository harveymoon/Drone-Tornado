"""
This Extension manages the MultiCam setup and calibration process.

"""

import cv2
import cv2.aruco as aruco
import numpy as np



class MultiCam:
	"""
	MultiCam description
	"""
	def __init__(self, ownerComp):
     
		print(f"OpenCV Version: {cv2.__version__}")
  
		self.ownerComp = ownerComp
		
		#board parameters
		self.squares_x = 8
		self.squares_y = 6
		self.square_length = 0.04
		self.marker_length = 0.02
  
		self.ChArDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
		self.ChArUco_board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length, self.marker_length, self.ChArDict)
		self.Parameters = cv2.aruco.DetectorParameters()
  
  
		return

	def GenerateCharucoBoard(self, width=800, height=600):
		"""
		Generates a ChArUco board image and saves it to the specified path.
		"""
		output_filename = "charuco_board.tiff"
		board_img = self.ChArUco_board.generateImage((width, height))
		cv2.imwrite(output_filename, board_img)
		print(f"ChArUco board saved as {output_filename}")
		return board_img

	def FindCharucoBoard(self, input_frame):
		"""
		Detects the ChArUco board in the input frame and returns the corners and IDs.
		"""
		print(f"Input frame shape: {input_frame.shape}, dtype: {input_frame.dtype}")
		inputFrameGray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
		outputFrame = input_frame.copy()

		corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(inputFrameGray, self.ChArDict, parameters=self.Parameters)
  
		foundArUco = {}

		foundArUco['corners'] = corners
		foundArUco['ids'] = ids

		# Check if any markers were detected
		if ids is not None and len(ids) > 0:
			print(f"Detected {len(ids)} ArUco markers")
			# Draw detected markers on the output image
			outputFrame = cv2.aruco.drawDetectedMarkers(outputFrame, corners, ids)
			# Refine the detected markers
			# This step helps improve the subpixel accuracy of marker corners
			for i in range(len(corners)):
					cv2.cornerSubPix(inputFrameGray, corners[i], 
								winSize=(3,3), 
								zeroZone=(-1,-1),
								criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
				# Find ChArUco corners
			retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, inputFrameGray, self.ChArUco_board)

			foundArUco['charucoCorners'] = charucoCorners
			foundArUco['charucoIds'] = charucoIds

			#print results	
			print(f"Found {len(corners)} ArUco corners")
			print(f"Found {len(ids)} ArUco IDs")
			print(f"Found {len(charucoCorners)} ChArUco corners")
			print(f"Found {len(charucoIds)} ChArUco IDs")

			# # If ChArUco corners were found, draw them on the output image
			# if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 0:
			# 	print(f"Found {len(charucoCorners)} ChArUco corners")
			# 	# Draw ChArUco corners
			# 	outputFrame = cv2.aruco.drawDetectedCornersCharuco(outputFrame, charucoCorners, charucoIds)
			# 	#  add corner numbers
			# 	for i in range(len(charucoIds)):
			# 		corner = tuple(charucoCorners[i][0].astype(int))
			# 		cv2.putText(outputFrame, str(charucoIds[i][0]), 
			# 					corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			# else:
			# 	print("No ChArUco corners found")
			return foundArUco
		else:
			print("No ArUco markers detected")
			return None
		
			

	def LoadTOP(self, top):
		"""
		Load a TOP image and convert it to a NumPy array.
		"""
		pixels = top.numpyArray(delayed=True)[:, :, :3] * 255.0
		input_frame = pixels.astype(np.uint8)
		input_frame = cv2.flip(input_frame, 0)
		return input_frame