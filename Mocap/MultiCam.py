"""
This Extension manages the MultiCam setup and calibration process.

"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json

def cv_pose_to_td(rvec, tvec,squares_x=8, squares_y=6, square_len=0.12):
	import cv2, numpy as np

	# --- boardTL → cam ------------------------------------------------------
	R, _ = cv2.Rodrigues(rvec)
	T_bc = np.eye(4)
	T_bc[:3,:3] = R
	T_bc[:3, 3] = tvec.flatten()

	# --- cam → boardTL ------------------------------------------------------
	T_cb = np.linalg.inv(T_bc)

	# --- move origin to board centre  (negative shift!) ---------------------
	half_w = 0.5 * (squares_x ) * square_len
	half_h = 0.5 * (squares_y ) * square_len
	T_tl_to_centre = np.eye(4)
	T_tl_to_centre[:3,3] = [-half_w, -half_h, 0]

	T_cb_centre = T_tl_to_centre @ T_cb      # NOTE: order

	# --- boardCentre → world  (fixed 90° about X, flip Z) -------------------
	board_to_world = np.array([
		[-1,  0,  0, 0],   # –X
		[ 0,  0,  -1, 0],   # +Y  (board Z up)
		[ 0, -1,  0, 0],   # –Z
		[ 0,  0,  0, 1]
	], dtype=np.float64)

	cam_to_world = board_to_world @ T_cb_centre
	return cam_to_world   # metres


def cv_to_gl_projection_matrix(K, w, h, znear=0.001, zfar=5000.0):
	# Construct an OpenGL-style projection matrix from an OpenCV-style projection
	# matrix
	#
	# References: 
	# [0] https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
	# [1] https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
	x0 = 0
	y0 = 0

	i00 = 2 * K[0, 0] / w
	i01 = -2 * K[0, 1] / w
	i02 = (w - 2 * K[0, 2] + 2 * x0) / w
	i03 = 0

	i10 = 0
	i11 = 2 * K[1, 1] / h
	i12 = (-h + 2 * K[1, 2] + 2 * y0) / h
	i13 = 0

	i20 = 0
	i21 = 0
	i22 = (-zfar - znear) / (zfar - znear)
	i23 = -2 * zfar * znear / (zfar - znear)

	i30 = 0
	i31 = 0
	i32 = -1
	i33 = 0

	return np.array([
		[i00, i01, i02, i03],
		[i10, i11, i12, i13],
		[i20, i21, i22, i23],
		[i30, i31, i32, i33]
	])


def pixel_to_world_ray(u, v, K, R, t):
    x_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])   # direction in cam coords
    x_cam /= np.linalg.norm(x_cam)

    # origin & dir in world coords
    dir_world = R.T @ x_cam
    origin_world = -R.T @ t
    return origin_world, dir_world

def triangulate_rays(origins, dirs):
    A = []
    b = []
    for o, d in zip(origins, dirs):
        d = d / np.linalg.norm(d)
        I = np.eye(3)
        A.append(I - np.outer(d, d))
        b.append((I - np.outer(d, d)) @ o)
    A = np.vstack(A)
    b = np.hstack(b)
    X, *_ = np.linalg.lstsq(A, b, rcond=None)
    return X          # 3‑D point in world coords


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
		self.square_length = 0.12
		self.marker_length = 0.07
  
		self.BoardSizeMeters = (self.squares_x * self.square_length, self.squares_y * self.square_length)
  
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
			if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 0:
				print(f"Found {len(charucoCorners)} ChArUco corners")
				print(f"Found {len(charucoIds)} ChArUco IDs")

			return foundArUco
		else:
			print("No ArUco markers detected")
			return None
		
	def TrinagulatePoint(self):
		"""tringulate a point from all found cameras"""
		c1u = op('base_simulation/null_blobs_c1')[1,'u'].val
		c1v = op('base_simulation/null_blobs_c1')[1,'v'].val

		c2u = op('base_simulation/null_blobs_c2')[1,'u'].val
		c2v = op('base_simulation/null_blobs_c2')[1,'v'].val

		c3u = op('base_simulation/null_blobs_c3')[1,'u'].val
		c3v = op('base_simulation/null_blobs_c3')[1,'v'].val

		c4u = op('base_simulation/null_blobs_c4')[1,'u'].val
		c4v = op('base_simulation/null_blobs_c4')[1,'v'].val

		p1 = [c1u, c1v]
		p2 = [c2u, c2v]
		p3 = [c3u, c3v]
		p4 = [c4u, c4v]
		# print(f"p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
    
		
    
	def LoadTOP(self, top, delay=False):
		"""
		Load a TOP image and convert it to a NumPy array.
		"""
		pixels = top.numpyArray(delayed = delay)[:, :, :3] * 255.0
		input_frame = pixels.astype(np.uint8)
		input_frame = cv2.flip(input_frame, 0)
		return input_frame

	def IntrinsicCalibration(self, imgFolder, save_path="intrinsics.json"):
		"""
		Calibrate a single camera from a folder of ChArUco‑board images.
		Returns (ret, K, dist, rvecs, tvecs)
		"""
		print("→ intrinsic calibration")

		# gather *.png (or *.tiff, *.jpg …)    
		image_files = [f for f in os.listdir(imgFolder)
					if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))]
		if not image_files:
			print("   no images in", imgFolder)
			return None

		all_corners, all_ids = [], []
		img_size = None

		for fname in image_files:
			img_path = os.path.join(imgFolder, fname)
			img = cv2.imread(img_path)
			if img is None:
				print("   couldn’t read", img_path)
				continue

			if img_size is None:
				img_size = img.shape[:2][::-1]   # (w,h)

			# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			res = self.FindCharucoBoard(img)
			if res is None:   # no markers
				continue

			char_corners = res["charucoCorners"]
			char_ids     = res["charucoIds"]

			if char_corners is not None and char_ids is not None and len(char_corners) > 3:
				all_corners.append(char_corners)
				all_ids.append(char_ids)
				print(f"   {fname}: {len(char_corners)} corners")
			else:
				print(f"   {fname}: not enough corners")

		if len(all_corners) < 5:
			print("   need at least 5 good views – got", len(all_corners))
			return None

		# calibrate
		ret, K, dist, rvecs, tvecs, *_ = cv2.aruco.calibrateCameraCharucoExtended(
			charucoCorners=all_corners,
			charucoIds=all_ids,
			board=self.ChArUco_board,
			imageSize=img_size,
			cameraMatrix=None,
			distCoeffs=None
		)

		print(f"   RMS reprojection error: {ret:.4f}")
		print("   camera matrix:\n", K)
		print("   dist coeffs:", dist.ravel())

		# stash for later
		# np.savez(save_path, K=K, dist=dist, rms=ret)
		outJSON = {
			"cameraMatrix": K.tolist(),
			"distCoeffs": dist.ravel().tolist(),
			"rms": ret
		}
		with open(save_path, 'w') as f:
			json.dump(outJSON, f, indent=4)
		print("   saved →", save_path)

		return ret, K, dist, rvecs, tvecs


	def Extrinsics_from_board(self, input_frame, calibrationData):
		"""
		Estimate camera pose w.r.t. the ChArUco board.
		
		Args
		----
		input_frame     : colour image (BGR or RGB – we detect channel count)
		calibrationData : dict or JSON‑loaded object with keys
						• 'cameraMatrix' 3×3
						• 'distCoeffs'   1×n  (n = 5, 8, …)
		Returns
		-------
		rvec (3,1), tvec (3,1), extrinsic 4×4  (or None if pose fails)
		"""
  
		targetMatrix = op('base_simulation/cam_sim1').worldTransform
		print(f'simulation camera matrix =  {targetMatrix}')
		
		print("→ extrinsic calibration")

		# --- load intrinsics ---
		K    = np.asarray(calibrationData['cameraMatrix'], dtype=np.float64)
		dist = np.asarray(calibrationData['distCoeffs'], dtype=np.float64)

		# --- detect board & interpolate ChArUco corners ---
		res = self.FindCharucoBoard(input_frame)
		if res is None:
			print("   no board found")
			return None

		char_corners = res['charucoCorners']
		char_ids     = res['charucoIds']
		if char_corners is None or len(char_corners) < 4:
			print("   not enough ChArUco corners")
			return None

		# --- pose estimation ---
		ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
			char_corners,              # detected charuco corners
			char_ids,                  # corresponding ids
			self.ChArUco_board,        # board object
			K,                         # intrinsics
			dist,                      # distortion
			None, None                 # ⟵ OUTPUT place‑holders
		)
		if not ok:
			print("   pose estimation failed")
			return None


		# --- build 4×4 extrinsic matrix ---
		R, _ = cv2.Rodrigues(rvec)
		extrinsic = np.eye(4, dtype=np.float64)
		extrinsic[:3, :3] = R
		extrinsic[:3,  3] = tvec.flatten()

		print("   rvec:", rvec.ravel())
		print("   tvec:", tvec.ravel())
		print("   extrinsic:\n", extrinsic)
  
		# targetNP = np.array(targetMatrix, dtype=np.float64)
  
		#check how close the result is to the target matrix
		if targetMatrix is not None:
			print("   target matrix:\n", targetMatrix)
			diff = np.abs(extrinsic - targetMatrix.numpyArray())
			# print("   difference:\n", diff)
			if np.all(diff < 0.01):
				print("   pose is close to target matrix")
			else:
				print("   pose is NOT close to target matrix")
  
  
		return rvec, tvec, extrinsic


	def ExtrinsicsToTable(self, rvec, tvec):
		"""
		Convert the extrinsics to a table format for TouchDesigner.
		Fixed version using correct coordinate transformation.
		"""
		# Use the fixed conversion function
		extrinsicSolve = cv_pose_to_td(rvec, tvec, 
									squares_x=self.squares_x, 
									squares_y=self.squares_y, 
									square_len=self.square_length)
		
		# Additional debug information
		print("   Fixed TD extrinsic matrix:\n", extrinsicSolve)
		
		# Convert to TD table format
		# extrinsic_flat = np.copy(extrinsicSolve).flatten()
		# tdu_extrinsic = tdu.Matrix(extrinsic_flat.tolist())
		tdu_extrinsic = tdu.Matrix(extrinsicSolve.flatten().tolist())
		# op('cam_prediction_1').setTransform(tdu_extrinsic)
  
		tdu_extrinsic.fillTable(op('table_cam_ext'))
		
		# Optional: Add validation against target matrix
		targetMatrix = op('base_simulation/cam_sim1').worldTransform
  
		print(f'simulation camera matrix =  {targetMatrix}')
		if targetMatrix is not None:
			targetNP = (targetMatrix).numpyArray()
			diff = np.abs(extrinsicSolve - targetNP)
			# print("   Difference after fix:\n", diff)
			print(f"   Max difference: {np.max(diff):.6f}")
			if np.all(diff < 0.1):  # More tolerant threshold
				print("   ✓ Pose is now close to target matrix!")
			else:
				print("   × Pose is still not close enough to target matrix")
		
		return extrinsicSolve

