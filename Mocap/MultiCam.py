"""
This Extension manages the MultiCam setup and calibration process.

"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json
import glob
from pathlib import Path

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
		[ 0,  0,  -1, 0],   # +Y  (board Z up)
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
  
		# Multi-camera calibration data
		self.cameras = {}  # Store camera calibration data
		self.world_poses = {}  # Store camera poses in world coordinates
		self.camera_relationships = {}  # Store relative poses between cameras
  
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
		# print(f"Input frame shape: {input_frame.shape}, dtype: {input_frame.dtype}")
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
			# print(f"Found {len(ids)} ArUco IDs")
			# print(f"Found {len(charucoCorners)} ChArUco corners")
			if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 0:
				print(f"Found {len(charucoCorners)} ChArUco corners")
				# print(f"Found {len(charucoIds)} ChArUco IDs")

			return foundArUco
		else:
			print("No ArUco markers detected")
			return None
		
	def LoadCameraIntrinsics(self, camera_folders=None):
		"""
		Load intrinsic calibration data for all cameras.
		
		Args:
			camera_folders: List of camera folder paths, or None to auto-detect
		
		Returns:
			dict: Camera intrinsics data
		"""
		if camera_folders is None:
			# Auto-detect camera folders
			frames_dir = Path("Frames")
			camera_folders = [f for f in frames_dir.iterdir() if f.is_dir() and f.name.startswith("Camera")]
			camera_folders.sort()
		
		self.cameras = {}
		
		for camera_folder in camera_folders:
			camera_name = camera_folder.name
			intrinsics_file = camera_folder / "intrinsics.json"
			
			if intrinsics_file.exists():
				with open(intrinsics_file, 'r') as f:
					intrinsics_data = json.load(f)
				
				self.cameras[camera_name] = {
					'intrinsics': intrinsics_data,
					'K': np.array(intrinsics_data['cameraMatrix'], dtype=np.float64),
					'dist': np.array(intrinsics_data['distCoeffs'], dtype=np.float64),
					'folder': camera_folder
				}
				print(f"Loaded intrinsics for {camera_name}")
			else:
				print(f"Warning: No intrinsics file found for {camera_name}")
		
		print(f"Loaded {len(self.cameras)} cameras")
		return self.cameras

	def CalibrateMultiCameraExtrinsics(self, reference_image_name="capture_3.tiff", save_results=True):
		"""
		Calibrate extrinsics for all cameras using a shared reference board image.
		
		Args:
			reference_image_name: Name of the image file showing the board (same across all cameras)
			save_results: Whether to save calibration results to files
		
		Returns:
			dict: World poses for each camera
		"""
		print("→ Multi-camera extrinsic calibration")
		
		if not self.cameras:
			self.LoadCameraIntrinsics()
		
		self.world_poses = {}
		reference_corners_3d = None
		reference_corners_ids = None
		
		# Process each camera
		for camera_name, camera_data in self.cameras.items():
			print(f"\n--- Processing {camera_name} ---")
			
			# Load the reference image
			image_path = camera_data['folder'] / reference_image_name
			if not image_path.exists():
				print(f"Warning: Reference image {reference_image_name} not found for {camera_name}")
				continue
			
			# Load and process image
			img = cv2.imread(str(image_path))
			if img is None:
				print(f"Error: Could not load image {image_path}")
				continue
			
			# Detect ChArUco board
			res = self.FindCharucoBoard(img)
			if res is None:
				print(f"No board detected in {camera_name}")
				continue
			
			char_corners = res['charucoCorners']
			char_ids = res['charucoIds']
			
			if char_corners is None or len(char_corners) < 4:
				print(f"Insufficient corners detected in {camera_name}")
				continue
			
			# Get 3D board points for detected corners
			obj_points = self.ChArUco_board.getChessboardCorners()
			
			# Match detected corner IDs to 3D points
			if char_ids is not None:
				# Create 3D points array for detected corners
				corners_3d = []
				corners_2d = []
				
				for i, corner_id in enumerate(char_ids.flatten()):
					if corner_id < len(obj_points):
						corners_3d.append(obj_points[corner_id])
						corners_2d.append(char_corners[i][0])  # char_corners is Nx1x2
				
				if len(corners_3d) < 4:
					print(f"Insufficient valid corners for {camera_name}")
					continue
				
				corners_3d = np.array(corners_3d, dtype=np.float64)
				corners_2d = np.array(corners_2d, dtype=np.float64)
				
				# Solve PnP to get camera pose
				success, rvec, tvec = cv2.solvePnP(
					corners_3d, corners_2d,
					camera_data['K'], camera_data['dist']
				)
				
				if success:
					# Convert to world pose (camera position in world coordinates)
					world_pose = cv_pose_to_td(rvec, tvec, self.squares_x, self.squares_y, self.square_length)
					
					self.world_poses[camera_name] = {
						'rvec': rvec,
						'tvec': tvec,
						'world_transform': world_pose,
						'corners_3d': corners_3d,
						'corners_2d': corners_2d
					}
					
					print(f"✓ Calibrated {camera_name}")
					print(f"  Position: {world_pose[:3, 3]}")
					print(f"  Corners detected: {len(corners_3d)}")
					
					# Save individual camera extrinsics
					if save_results:
						self.SaveCameraExtrinsics(camera_name, world_pose, rvec, tvec)
				else:
					print(f"✗ PnP failed for {camera_name}")
		
		# Calculate camera relationships
		self.CalculateCameraRelationships()
		
		# Save multi-camera calibration summary
		if save_results:
			self.SaveMultiCameraCalibration()
		
		print(f"\n✓ Multi-camera calibration complete: {len(self.world_poses)} cameras calibrated")
		return self.world_poses

	def SaveCameraExtrinsics(self, camera_name, world_transform, rvec, tvec):
		"""Save individual camera extrinsics to file."""
		camera_folder = self.cameras[camera_name]['folder']
		extrinsics_file = camera_folder / "extrinsics.json"
		
		extrinsics_data = {
			'world_transform': world_transform.tolist(),
			'rvec': rvec.tolist(),
			'tvec': tvec.tolist(),
			'board_parameters': {
				'squares_x': self.squares_x,
				'squares_y': self.squares_y,
				'square_length': self.square_length,
				'marker_length': self.marker_length
			}
		}
		
		with open(extrinsics_file, 'w') as f:
			json.dump(extrinsics_data, f, indent=4)
		
		print(f"  Saved extrinsics to {extrinsics_file}")

	def CalculateCameraRelationships(self):
		"""Calculate relative poses between all camera pairs."""
		self.camera_relationships = {}
		
		camera_names = list(self.world_poses.keys())
		
		for i, cam1 in enumerate(camera_names):
			for j, cam2 in enumerate(camera_names):
				if i != j:
					# Calculate relative transform from cam1 to cam2
					T1 = self.world_poses[cam1]['world_transform']
					T2 = self.world_poses[cam2]['world_transform']
					
					# T_rel = T2 * inv(T1)  (transform from cam1 to cam2)
					T_rel = T2 @ np.linalg.inv(T1)
					
					pair_name = f"{cam1}_to_{cam2}"
					self.camera_relationships[pair_name] = {
						'transform': T_rel,
						'translation': T_rel[:3, 3],
						'rotation': T_rel[:3, :3],
						'distance': np.linalg.norm(T_rel[:3, 3])
					}
		
		print(f"Calculated {len(self.camera_relationships)} camera relationships")

	def SaveMultiCameraCalibration(self):
		"""Save complete multi-camera calibration data."""
		calibration_data = {
			'cameras': {},
			'relationships': {},
			'board_parameters': {
				'squares_x': self.squares_x,
				'squares_y': self.squares_y,
				'square_length': self.square_length,
				'marker_length': self.marker_length
			}
		}
		
		# Save camera poses
		for camera_name, pose_data in self.world_poses.items():
			calibration_data['cameras'][camera_name] = {
				'world_transform': pose_data['world_transform'].tolist(),
				'position': pose_data['world_transform'][:3, 3].tolist(),
				'rvec': pose_data['rvec'].tolist(),
				'tvec': pose_data['tvec'].tolist()
			}
		
		# Save relationships
		for pair_name, rel_data in self.camera_relationships.items():
			calibration_data['relationships'][pair_name] = {
				'transform': rel_data['transform'].tolist(),
				'translation': rel_data['translation'].tolist(),
				'distance': float(rel_data['distance'])
			}
		
		# Save to file
		with open("multicamera_calibration.json", 'w') as f:
			json.dump(calibration_data, f, indent=4)
		
		print("Saved complete multi-camera calibration to multicamera_calibration.json")

	def LoadMultiCameraCalibration(self, filename="multicamera_calibration.json"):
		"""Load previously saved multi-camera calibration."""
		if not os.path.exists(filename):
			print(f"Calibration file {filename} not found")
			return False
		
		with open(filename, 'r') as f:
			calibration_data = json.load(f)
		
		# Load camera poses
		self.world_poses = {}
		for camera_name, cam_data in calibration_data['cameras'].items():
			self.world_poses[camera_name] = {
				'world_transform': np.array(cam_data['world_transform']),
				'rvec': np.array(cam_data['rvec']),
				'tvec': np.array(cam_data['tvec'])
			}
		
		# Load relationships
		self.camera_relationships = {}
		for pair_name, rel_data in calibration_data['relationships'].items():
			self.camera_relationships[pair_name] = {
				'transform': np.array(rel_data['transform']),
				'translation': np.array(rel_data['translation']),
				'distance': rel_data['distance']
			}
		
		print(f"Loaded multi-camera calibration: {len(self.world_poses)} cameras")
		return True

	def TriangulateMultiCamera(self, pixel_coordinates):
		"""
		Triangulate a 3D point from multiple camera observations.
		
		Args:
			pixel_coordinates: dict with camera_name: [u, v] pixel coordinates
		
		Returns:
			np.array: 3D world coordinates [x, y, z]
		"""
		if not self.cameras or not self.world_poses:
			print("Error: Camera calibration data not loaded")
			return None
		
		origins = []
		directions = []
		
		for camera_name, (u, v) in pixel_coordinates.items():
			if camera_name not in self.cameras or camera_name not in self.world_poses:
				print(f"Warning: {camera_name} not calibrated, skipping")
				continue
			
			# Get camera parameters
			K = self.cameras[camera_name]['K']
			dist = self.cameras[camera_name]['dist']
			world_transform = self.world_poses[camera_name]['world_transform']
			
			# Extract rotation and translation from world transform
			R = world_transform[:3, :3]
			t = world_transform[:3, 3]
			
			# Convert pixel to ray in world coordinates
			origin, direction = pixel_to_world_ray(u, v, K, R, t)
			
			origins.append(origin)
			directions.append(direction)
		
		if len(origins) < 2:
			print("Error: Need at least 2 cameras for triangulation")
			return None
		
		# Triangulate using least squares
		point_3d = triangulate_rays(origins, directions)
		
		return point_3d

	def TriangulateFromTouchDesigner(self):
		"""
		Triangulate a point using blob coordinates from TouchDesigner operators.
		This is the updated version of your existing TrinagulatePoint method.
		"""
		try:
			# Get pixel coordinates from TouchDesigner operators
			pixel_coords = {}
			
			# Check if operators exist and get coordinates
			cameras_to_check = [
				('Camera1', 'base_simulation/null_blobs_c1'),
				('Camera2', 'base_simulation/null_blobs_c2'),
				('Camera3', 'base_simulation/null_blobs_c3'),
				('Camera4', 'base_simulation/null_blobs_c4')
			]
			
			for camera_name, op_path in cameras_to_check:
				try:
					blob_op = op(op_path)
					if blob_op and blob_op.numRows > 1:  # Check if blob data exists
						u = blob_op[1, 'u'].val
						v = blob_op[1, 'v'].val
						pixel_coords[camera_name] = [u, v]
						print(f"{camera_name}: ({u:.1f}, {v:.1f})")
				except:
					print(f"Warning: Could not get blob data from {op_path}")
			
			if len(pixel_coords) < 2:
				print("Error: Need at least 2 cameras with blob data")
				return None
			
			# Triangulate the 3D point
			point_3d = self.TriangulateMultiCamera(pixel_coords)
			
			if point_3d is not None:
				print(f"Triangulated 3D point: [{point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}]")
				
				# Optionally store result in a TouchDesigner table
				try:
					result_table = op('table_triangulated_point')
					if result_table:
						result_table.clear()
						result_table.appendRow(['x', 'y', 'z'])
						result_table.appendRow([point_3d[0], point_3d[1], point_3d[2]])
				except:
					pass  # Table doesn't exist, that's okay
			
			return point_3d
			
		except Exception as e:
			print(f"Error in triangulation: {e}")
			return None

	def ValidateCalibration(self, test_image_name="capture_10.tiff"):
		"""
		Validate multi-camera calibration by checking reprojection errors.
		
		Args:
			test_image_name: Name of test image to validate against
		"""
		print("→ Validating multi-camera calibration")
		
		if not self.cameras or not self.world_poses:
			print("Error: No calibration data loaded")
			return
		
		total_error = 0
		camera_count = 0
		validation_results = {}
		
		for camera_name, camera_data in self.cameras.items():
			if camera_name not in self.world_poses:
				print(f"⚠️  {camera_name}: No calibration data available")
				continue
			
			# Load test image
			image_path = camera_data['folder'] / test_image_name
			if not image_path.exists():
				print(f"⚠️  {camera_name}: Test image {test_image_name} not found")
				continue
			
			img = cv2.imread(str(image_path))
			if img is None:
				print(f"⚠️  {camera_name}: Could not load test image")
				continue
			
			# Detect board
			res = self.FindCharucoBoard(img)
			if res is None:
				print(f"⚠️  {camera_name}: No board detected in test image")
				continue
			
			char_corners = res['charucoCorners']
			char_ids = res['charucoIds']
			
			if char_corners is None or len(char_corners) < 4:
				print(f"⚠️  {camera_name}: Insufficient corners detected ({len(char_corners) if char_corners is not None else 0})")
				continue
			
			# Get 3D points and reproject
			obj_points = self.ChArUco_board.getChessboardCorners()
			corners_3d = []
			corners_2d = []
			
			for i, corner_id in enumerate(char_ids.flatten()):
				if corner_id < len(obj_points):
					corners_3d.append(obj_points[corner_id])
					corners_2d.append(char_corners[i][0])
			
			if len(corners_3d) < 4:
				print(f"⚠️  {camera_name}: Insufficient valid corners ({len(corners_3d)})")
				continue
			
			corners_3d = np.array(corners_3d, dtype=np.float64)
			corners_2d = np.array(corners_2d, dtype=np.float64)
			
			# Reproject using calibrated pose
			try:
				rvec = self.world_poses[camera_name]['rvec']
				tvec = self.world_poses[camera_name]['tvec']
				
				projected_points, _ = cv2.projectPoints(
					corners_3d, rvec, tvec,
					camera_data['K'], camera_data['dist']
				)
				
				# Calculate reprojection error
				projected_points = projected_points.reshape(-1, 2)
				errors = np.sqrt(np.sum((corners_2d - projected_points) ** 2, axis=1))
				error = np.mean(errors)
				max_error = np.max(errors)
				
				validation_results[camera_name] = {
					'error': error,
					'max_error': max_error,
					'corners_used': len(corners_3d),
					'status': 'success'
				}
				
				print(f"✓ {camera_name}: Reprojection error = {error:.3f} pixels (max: {max_error:.3f}, corners: {len(corners_3d)})")
				
				# Check if error is reasonable
				if error > 100:  # Very high error threshold
					print(f"  ⚠️  WARNING: Very high reprojection error for {camera_name}")
					validation_results[camera_name]['status'] = 'high_error'
				elif error > 10:
					print(f"  ⚠️  WARNING: High reprojection error for {camera_name}")
					validation_results[camera_name]['status'] = 'moderate_error'
				
				total_error += error
				camera_count += 1
				
			except Exception as e:
				print(f"❌ {camera_name}: Validation failed - {str(e)}")
				validation_results[camera_name] = {
					'error': float('inf'),
					'status': 'failed',
					'error_message': str(e)
				}
		
		# Summary
		if camera_count > 0:
			avg_error = total_error / camera_count
			print(f"\n📊 Validation Summary:")
			print(f"• Cameras tested: {camera_count}/{len(self.world_poses)}")
			print(f"• Average reprojection error: {avg_error:.3f} pixels")
			
			if avg_error < 1.0:
				print("✅ Calibration quality: Excellent")
			elif avg_error < 2.0:
				print("✅ Calibration quality: Good")
			elif avg_error < 5.0:
				print("⚠️  Calibration quality: Fair")
			elif avg_error < 20.0:
				print("❌ Calibration quality: Poor - consider recalibrating")
			else:
				print("❌ Calibration quality: Very Poor - recalibration required")
		else:
			print("❌ No cameras could be validated")
		
		return validation_results

	def GetCameraPositions(self):
		"""
		Get the positions of all calibrated cameras in world coordinates.
		
		Returns:
			dict: Camera positions {camera_name: [x, y, z]}
		"""
		positions = {}
		for camera_name, pose_data in self.world_poses.items():
			position = pose_data['world_transform'][:3, 3]
			positions[camera_name] = position.tolist()
		
		return positions

	def PrintCalibrationSummary(self):
		"""Print a summary of the multi-camera calibration."""
		print("\n" + "="*50)
		print("MULTI-CAMERA CALIBRATION SUMMARY")
		print("="*50)
		
		if not self.cameras:
			print("No cameras loaded")
			return
		
		print(f"Cameras loaded: {len(self.cameras)}")
		print(f"Cameras calibrated: {len(self.world_poses)}")
		
		if self.world_poses:
			print("\nCamera Positions (world coordinates):")
			for camera_name, pose_data in self.world_poses.items():
				pos = pose_data['world_transform'][:3, 3]
				print(f"  {camera_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
		
		if self.camera_relationships:
			print(f"\nCamera relationships calculated: {len(self.camera_relationships)}")
			print("Inter-camera distances:")
			for pair_name, rel_data in self.camera_relationships.items():
				if "_to_" in pair_name:
					cam1, cam2 = pair_name.split("_to_")
					if cam1 < cam2:  # Only print each pair once
						distance = rel_data['distance']
						print(f"  {cam1} ↔ {cam2}: {distance:.3f}m")
		
		print("="*50)

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
		min_corners_required = 6  # DLT algorithm needs at least 6 points

		for fname in image_files:
			img_path = os.path.join(imgFolder, fname)
			img = cv2.imread(img_path)
			if img is None:
				print("   couldn't read", img_path)
				continue

			if img_size is None:
				img_size = img.shape[:2][::-1]   # (w,h)

			# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			res = self.FindCharucoBoard(img)
			if res is None:   # no markers
				print(f"   {fname}: no markers detected")
				continue

			char_corners = res["charucoCorners"]
			char_ids     = res["charucoIds"]

			if char_corners is not None and char_ids is not None and len(char_corners) >= min_corners_required:
				all_corners.append(char_corners)
				all_ids.append(char_ids)
				print(f"   {fname}: {len(char_corners)} corners (✓)")
			else:
				corner_count = len(char_corners) if char_corners is not None else 0
				print(f"   {fname}: {corner_count} corners (need ≥{min_corners_required}) (✗)")

		print(f"   Total valid images: {len(all_corners)}")
		
		if len(all_corners) < 5:
			print(f"   ERROR: need at least 5 good views – got {len(all_corners)}")
			print(f"   Each image needs ≥{min_corners_required} ChArUco corners")
			print("   Try:")
			print("     • Ensure the ChArUco board is fully visible in images")
			print("     • Check lighting conditions")
			print("     • Verify board parameters match the physical board")
			return None

		# Additional validation: check total corner count across all images
		total_corners = sum(len(corners) for corners in all_corners)
		print(f"   Total corners across all images: {total_corners}")

		# Check for pose diversity to avoid degenerate configurations
		corner_positions = []
		for corners in all_corners:
			if corners is not None and len(corners) > 0:
				# Get center of detected corners for this image
				center = np.mean(corners.reshape(-1, 2), axis=0)
				corner_positions.append(center)
		
		if len(corner_positions) > 1:
			corner_positions = np.array(corner_positions)
			# Check spread of board positions across images
			pos_std = np.std(corner_positions, axis=0)
			print(f"   Board position diversity (std): x={pos_std[0]:.1f}, y={pos_std[1]:.1f} pixels")
			
			if pos_std[0] < 50 or pos_std[1] < 50:
				print("   WARNING: Low pose diversity detected!")
				print("   Consider taking images with the board at different:")
				print("     • Distances from camera")
				print("     • Orientations (tilted, rotated)")
				print("     • Positions across the image")

		# Diagnostic: Check board parameters and corner IDs
		print("   Running diagnostics...")
		
		# Check expected vs actual board size
		expected_corners = (self.squares_x - 1) * (self.squares_y - 1)  # 7 * 5 = 35
		print(f"   Expected max ChArUco corners: {expected_corners}")
		
		# Analyze corner ID distribution
		all_corner_ids = []
		for ids in all_ids:
			if ids is not None:
				all_corner_ids.extend(ids.flatten())
		
		unique_ids = np.unique(all_corner_ids)
		print(f"   Unique corner IDs detected: {len(unique_ids)} (range: {min(unique_ids)}-{max(unique_ids)})")
		
		if len(unique_ids) < 10:
			print("   WARNING: Very few unique corner IDs detected!")
			print("   This suggests board parameter mismatch or poor detection")
		
		# Check corner distribution per image
		corners_per_image = [len(corners) for corners in all_corners]
		avg_corners = np.mean(corners_per_image)
		print(f"   Average corners per image: {avg_corners:.1f}")
		
		if avg_corners < 10:
			print("   WARNING: Low average corners per image")
			print("   Consider:")
			print("     • Moving camera closer to board")
			print("     • Using better lighting")
			print("     • Checking if board parameters match physical board")

		# Try different calibration approaches
		calibration_attempts = [
			("Standard ChArUco", lambda: cv2.aruco.calibrateCameraCharucoExtended(
				all_corners, all_ids, self.ChArUco_board, img_size, None, None, flags=0)),
			("ChArUco with CALIB_FIX_ASPECT_RATIO", lambda: cv2.aruco.calibrateCameraCharucoExtended(
				all_corners, all_ids, self.ChArUco_board, img_size, None, None, 
				flags=cv2.CALIB_FIX_ASPECT_RATIO)),
			("ChArUco with CALIB_ZERO_TANGENT_DIST", lambda: cv2.aruco.calibrateCameraCharucoExtended(
				all_corners, all_ids, self.ChArUco_board, img_size, None, None, 
				flags=cv2.CALIB_ZERO_TANGENT_DIST)),
			("ChArUco with simplified distortion", lambda: cv2.aruco.calibrateCameraCharucoExtended(
				all_corners, all_ids, self.ChArUco_board, img_size, None, None, 
				flags=cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6))
		]

		ret, K, dist, rvecs, tvecs = None, None, None, None, None
		
		for method_name, calibration_func in calibration_attempts:
			print(f"   Attempting: {method_name}")
			try:
				result = calibration_func()
				ret, K, dist, rvecs, tvecs = result[:5]
				print(f"   ✓ {method_name} succeeded!")
				break
			except cv2.error as e:
				print(f"   ✗ {method_name} failed: {str(e)[:100]}...")
				continue
		
		if ret is None:
			print("   ERROR: All calibration methods failed!")
			print("   Possible issues:")
			print("     • Board parameters don't match physical board")
			print("     • Board is not flat/rigid")
			print("     • Corner detection quality is poor")
			print("     • Images are corrupted or low quality")
			print("   ")
			print("   Debug suggestions:")
			print("     • Verify board dimensions with ruler")
			print("     • Check that squares_x=8, squares_y=6 match your board")
			print("     • Ensure square_length=0.12m and marker_length=0.07m are correct")
			print("     • Try with fewer, higher-quality images")
			return None

		print(f"   RMS reprojection error: {ret:.4f}")
		print("   camera matrix:\n", K)
		print("   dist coeffs:", dist.ravel())

		# Validate calibration results
		if ret > 1.0:
			print(f"   WARNING: High reprojection error ({ret:.4f})")
			print("   Consider:")
			print("     • Taking more/better calibration images")
			print("     • Checking board flatness")
			print("     • Verifying board dimensions")

		# stash for later
		# np.savez(save_path, K=K, dist=dist, rms=ret)
		outJSON = {
			"cameraMatrix": K.tolist(),
			"distCoeffs": dist.ravel().tolist(),
			"rms": ret
		}
		with open(imgFolder +'/'+ save_path, 'w') as f:
			json.dump(outJSON, f, indent=4)
		print("   saved →", imgFolder +'/'+ save_path)

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
			None, None                 # ⟵ OUTPUT place‑holders
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

	def ExtrinsicsToTable(self, rvec, tvec, outDat):
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
		
		tdu_extrinsic.fillTable(outDat)
		
		# # Optional: Add validation against target matrix
		# targetMatrix = op('base_simulation/cam_sim1').worldTransform
		
		# print(f'simulation camera matrix =  {targetMatrix}')
		# if targetMatrix is not None:
		# 	targetNP = (targetMatrix).numpyArray()
		# 	diff = np.abs(extrinsicSolve - targetNP)
		# 	# print("   Difference after fix:\n", diff)
		# 	print(f"   Max difference: {np.max(diff):.6f}")
		# 	if np.all(diff < 0.1):  # More tolerant threshold
		# 		print("   ✓ Pose is now close to target matrix!")
		# 	else:
		# 		print("   × Pose is still not close enough to target matrix")
		
		return extrinsicSolve

	def CollectSharedFeatures(self, image_names=None, min_shared_corners=10):
		"""
		Collect shared ChArUco corner features across multiple cameras.
		
		Args:
			image_names: List of image names to process, or None for auto-detection
			min_shared_corners: Minimum shared corners required between camera pairs
		
		Returns:
			dict: Shared features data structure
		"""
		print("→ Collecting shared features across cameras")
		
		if not self.cameras:
			self.LoadCameraIntrinsics()
		
		if image_names is None:
			# Auto-detect common images across all cameras
			all_images = set()
			for camera_data in self.cameras.values():
				folder = camera_data['folder']
				images = [f.name for f in folder.glob("capture_*.tiff")]
				if not all_images:
					all_images = set(images)
				else:
					all_images = all_images.intersection(set(images))
			
			# Use more images - up to 100 instead of 20
			image_names = sorted(list(all_images))[:100]
			print(f"Found {len(all_images)} common images across all cameras")
			print(f"Using {len(image_names)} images for analysis")
		
		shared_features = {
			'images': [],
			'camera_corners': {},  # camera_name -> [image_corners_list]
			'camera_ids': {},      # camera_name -> [image_ids_list]
			'shared_corner_ids': [],  # corner IDs visible in multiple cameras
			'image_names': image_names,
			'detection_stats': {}  # Statistics about detection quality
		}
		
		# Initialize storage for each camera
		for camera_name in self.cameras.keys():
			shared_features['camera_corners'][camera_name] = []
			shared_features['camera_ids'][camera_name] = []
			shared_features['detection_stats'][camera_name] = {
				'total_detections': 0,
				'good_detections': 0,
				'corner_counts': []
			}
		
		# Process each image
		valid_images = 0
		for img_idx, img_name in enumerate(image_names):
			if img_idx % 10 == 0:  # Progress update every 10 images
				print(f"Processing images {img_idx+1}-{min(img_idx+10, len(image_names))} of {len(image_names)}")
			
			image_data = {
				'name': img_name,
				'cameras': {},
				'shared_corners': {}
			}
			
			# Detect corners in each camera for this image
			cameras_with_detection = 0
			for camera_name, camera_data in self.cameras.items():
				image_path = camera_data['folder'] / img_name
				if not image_path.exists():
					continue
				
				img = cv2.imread(str(image_path))
				if img is None:
					continue
				
				# Detect ChArUco board
				res = self.FindCharucoBoard(img)
				shared_features['detection_stats'][camera_name]['total_detections'] += 1
				
				if res is None:
					continue
				
				char_corners = res['charucoCorners']
				char_ids = res['charucoIds']
				
				if char_corners is not None and char_ids is not None and len(char_corners) >= 4:
					# Store corners and IDs for this camera/image
					corners_2d = char_corners.reshape(-1, 2)
					ids_flat = char_ids.flatten()
					
					image_data['cameras'][camera_name] = {
						'corners': corners_2d,
						'ids': ids_flat
					}
					
					shared_features['camera_corners'][camera_name].append(corners_2d)
					shared_features['camera_ids'][camera_name].append(ids_flat)
					shared_features['detection_stats'][camera_name]['good_detections'] += 1
					shared_features['detection_stats'][camera_name]['corner_counts'].append(len(corners_2d))
					cameras_with_detection += 1
			
			# Find shared corner IDs across cameras for this image
			if len(image_data['cameras']) >= 2:
				camera_names = list(image_data['cameras'].keys())
				shared_ids = set(image_data['cameras'][camera_names[0]]['ids'])
				
				for camera_name in camera_names[1:]:
					shared_ids = shared_ids.intersection(set(image_data['cameras'][camera_name]['ids']))
				
				# Be more flexible with shared corners requirement
				min_corners_for_image = max(4, min_shared_corners // 2)  # At least 4, but allow lower threshold
				
				if len(shared_ids) >= min_corners_for_image:
					image_data['shared_corner_ids'] = list(shared_ids)
					shared_features['images'].append(image_data)
					valid_images += 1
					if img_idx < 20:  # Only print details for first 20 images
						print(f"  ✓ {img_name}: {len(shared_ids)} shared corners across {len(image_data['cameras'])} cameras")
				else:
					if img_idx < 20:  # Only print details for first 20 images
						print(f"  ✗ {img_name}: Only {len(shared_ids)} shared corners (need ≥{min_corners_for_image})")
			else:
				if img_idx < 20:  # Only print details for first 20 images
					print(f"  ✗ {img_name}: Only {len(image_data['cameras'])} cameras detected")
		
		print(f"\nCollection Summary:")
		print(f"• Processed {len(image_names)} images")
		print(f"• Found {valid_images} images with sufficient shared features")
		
		# Print detection statistics
		print(f"\nDetection Statistics:")
		for camera_name, stats in shared_features['detection_stats'].items():
			total = stats['total_detections']
			good = stats['good_detections']
			rate = (good / total * 100) if total > 0 else 0
			avg_corners = np.mean(stats['corner_counts']) if stats['corner_counts'] else 0
			print(f"  {camera_name}: {good}/{total} ({rate:.1f}%) - Avg corners: {avg_corners:.1f}")
		
		return shared_features

	def StereoCalibratePair(self, camera1_name, camera2_name, shared_features):
		"""
		Perform stereo calibration between two cameras using shared features.
		
		Args:
			camera1_name, camera2_name: Names of the two cameras
			shared_features: Shared features data from CollectSharedFeatures
		
		Returns:
			dict: Stereo calibration results
		"""
		print(f"→ Stereo calibrating {camera1_name} ↔ {camera2_name}")
		
		if camera1_name not in self.cameras or camera2_name not in self.cameras:
			print("Error: One or both cameras not found")
			return None
		
		# Get camera intrinsics
		K1 = self.cameras[camera1_name]['K']
		dist1 = self.cameras[camera1_name]['dist']
		K2 = self.cameras[camera2_name]['K']
		dist2 = self.cameras[camera2_name]['dist']
		
		# Collect corresponding points
		object_points = []
		image_points1 = []
		image_points2 = []
		
		obj_corners = self.ChArUco_board.getChessboardCorners()
		
		for image_data in shared_features['images']:
			if camera1_name in image_data['cameras'] and camera2_name in image_data['cameras']:
				cam1_data = image_data['cameras'][camera1_name]
				cam2_data = image_data['cameras'][camera2_name]
				
				# Find shared corner IDs between these two cameras
				shared_ids = set(cam1_data['ids']).intersection(set(cam2_data['ids']))
				
				if len(shared_ids) >= 8:  # Minimum for stereo calibration
					# Extract corresponding points
					obj_pts = []
					img_pts1 = []
					img_pts2 = []
					
					for corner_id in shared_ids:
						if corner_id < len(obj_corners):
							# Find indices in each camera's arrays
							idx1 = np.where(cam1_data['ids'] == corner_id)[0]
							idx2 = np.where(cam2_data['ids'] == corner_id)[0]
							
							if len(idx1) > 0 and len(idx2) > 0:
								obj_pts.append(obj_corners[corner_id])
								img_pts1.append(cam1_data['corners'][idx1[0]])
								img_pts2.append(cam2_data['corners'][idx2[0]])
					
					if len(obj_pts) >= 8:
						object_points.append(np.array(obj_pts, dtype=np.float32))
						image_points1.append(np.array(img_pts1, dtype=np.float32))
						image_points2.append(np.array(img_pts2, dtype=np.float32))
		
		if len(object_points) < 5:
			print(f"Error: Not enough valid image pairs ({len(object_points)} < 5)")
			return None
		
		print(f"Using {len(object_points)} image pairs for stereo calibration")
		
		# Get image size (assume all images same size)
		img_size = None
		for camera_data in self.cameras.values():
			sample_img_path = list(camera_data['folder'].glob("capture_*.tiff"))[0]
			sample_img = cv2.imread(str(sample_img_path))
			if sample_img is not None:
				img_size = sample_img.shape[:2][::-1]  # (width, height)
				break
		
		if img_size is None:
			print("Error: Could not determine image size")
			return None
		
		# Perform stereo calibration
		try:
			flags = (cv2.CALIB_FIX_INTRINSIC |  # Don't change intrinsics
					cv2.CALIB_RATIONAL_MODEL |   # Use rational distortion model
					cv2.CALIB_FIX_PRINCIPAL_POINT)  # Fix principal points
			
			ret, K1_new, dist1_new, K2_new, dist2_new, R, T, E, F = cv2.stereoCalibrate(
				object_points, image_points1, image_points2,
				K1, dist1, K2, dist2, img_size,
				flags=flags,
				criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
			)
			
			print(f"Stereo calibration RMS error: {ret:.4f}")
			
			# Calculate relative pose
			baseline = np.linalg.norm(T)
			
			stereo_result = {
				'camera1': camera1_name,
				'camera2': camera2_name,
				'rms_error': ret,
				'R': R,  # Rotation from camera1 to camera2
				'T': T,  # Translation from camera1 to camera2
				'E': E,  # Essential matrix
				'F': F,  # Fundamental matrix
				'baseline': baseline,
				'num_images': len(object_points)
			}
			
			print(f"Baseline distance: {baseline:.3f}m")
			return stereo_result
			
		except cv2.error as e:
			print(f"Stereo calibration failed: {e}")
			return None

	def CalibrateMultiCameraStereo(self, reference_camera='Camera1', save_results=True):
		"""
		Perform multi-camera calibration using stereo calibration pairs.
		
		Args:
			reference_camera: Camera to use as world coordinate reference
			save_results: Whether to save results
		
		Returns:
			dict: Multi-camera calibration results
		"""
		print("→ Multi-camera stereo calibration")
		
		if not self.cameras:
			self.LoadCameraIntrinsics()
		
		# Step 1: Collect shared features with more flexible requirements
		print("Collecting shared features with flexible requirements...")
		shared_features = self.CollectSharedFeatures(min_shared_corners=6)  # Lower threshold
		
		if len(shared_features['images']) < 3:
			print("⚠️  Not enough images with shared features for stereo calibration")
			print("   Falling back to simple board-based calibration...")
			return self.CalibrateMultiCameraExtrinsics("capture_3.tiff", save_results)
		
		# Step 2: Analyze which camera pairs have enough data
		camera_names = list(self.cameras.keys())
		pair_analysis = {}
		
		print(f"\nAnalyzing camera pair viability:")
		for i, cam1 in enumerate(camera_names):
			for j, cam2 in enumerate(camera_names):
				if i < j:  # Only check each pair once
					pair_key = f"{cam1}_{cam2}"
					
					# Count valid images for this pair
					valid_pairs = 0
					total_shared_corners = 0
					
					for image_data in shared_features['images']:
						if cam1 in image_data['cameras'] and cam2 in image_data['cameras']:
							cam1_data = image_data['cameras'][cam1]
							cam2_data = image_data['cameras'][cam2]
							shared_ids = set(cam1_data['ids']).intersection(set(cam2_data['ids']))
							
							if len(shared_ids) >= 6:  # Minimum for this pair
								valid_pairs += 1
								total_shared_corners += len(shared_ids)
					
					avg_corners = total_shared_corners / valid_pairs if valid_pairs > 0 else 0
					pair_analysis[pair_key] = {
						'cam1': cam1,
						'cam2': cam2,
						'valid_pairs': valid_pairs,
						'avg_corners': avg_corners,
						'viable': valid_pairs >= 3  # Need at least 3 good pairs
					}
					
					status = "✓" if pair_analysis[pair_key]['viable'] else "✗"
					print(f"  {status} {cam1} ↔ {cam2}: {valid_pairs} valid pairs, {avg_corners:.1f} avg corners")
		
		# Step 3: Find the best calibration strategy
		viable_pairs = {k: v for k, v in pair_analysis.items() if v['viable']}
		
		if not viable_pairs:
			print("❌ No viable camera pairs found for stereo calibration")
			print("   Falling back to simple board-based calibration...")
			return self.CalibrateMultiCameraExtrinsics("capture_3.tiff", save_results)
		
		print(f"\nFound {len(viable_pairs)} viable camera pairs")
		
		# Step 4: Perform stereo calibrations
		stereo_results = {}
		
		# Strategy 1: Try to calibrate reference camera with others
		reference_pairs = [pair for pair in viable_pairs.values() 
						  if pair['cam1'] == reference_camera or pair['cam2'] == reference_camera]
		
		if reference_pairs:
			print(f"\nCalibrating {reference_camera} with other cameras:")
			for pair_data in reference_pairs:
				cam1, cam2 = pair_data['cam1'], pair_data['cam2']
				pair_key = f"{cam1}_{cam2}"
				
				stereo_result = self.StereoCalibratePair(cam1, cam2, shared_features)
				if stereo_result is not None:
					stereo_results[pair_key] = stereo_result
					print(f"  ✓ {cam1} ↔ {cam2} calibrated successfully")
				else:
					print(f"  ✗ {cam1} ↔ {cam2} calibration failed")
		
		# Strategy 2: If reference camera pairs failed, try other pairs
		if not stereo_results:
			print(f"\nReference camera pairs failed, trying other combinations:")
			for pair_key, pair_data in viable_pairs.items():
				if pair_key not in stereo_results:
					cam1, cam2 = pair_data['cam1'], pair_data['cam2']
					stereo_result = self.StereoCalibratePair(cam1, cam2, shared_features)
					if stereo_result is not None:
						stereo_results[pair_key] = stereo_result
						print(f"  ✓ {cam1} ↔ {cam2} calibrated successfully")
						# Use first successful camera as reference
						reference_camera = cam1
						break
					else:
						print(f"  ✗ {cam1} ↔ {cam2} calibration failed")
		
		if not stereo_results:
			print("❌ All stereo calibrations failed")
			print("   Falling back to simple board-based calibration...")
			return self.CalibrateMultiCameraExtrinsics("capture_3.tiff", save_results)
		
		# Step 5: Build world poses from successful stereo results
		self.world_poses = {}
		
		# Reference camera at origin
		self.world_poses[reference_camera] = {
			'world_transform': np.eye(4, dtype=np.float64),
			'rvec': np.zeros((3, 1)),
			'tvec': np.zeros((3, 1)),
			'method': 'reference'
		}
		
		print(f"\nUsing {reference_camera} as reference camera")
		
		# Position other cameras relative to reference
		positioned_cameras = {reference_camera}
		
		for pair_key, stereo_result in stereo_results.items():
			cam1, cam2 = stereo_result['camera1'], stereo_result['camera2']
			
			# Determine which camera to position
			if cam1 == reference_camera and cam2 not in positioned_cameras:
				# Position cam2 relative to cam1 (reference)
				target_camera = cam2
				R = stereo_result['R']
				T = stereo_result['T'].reshape(3, 1)
			elif cam2 == reference_camera and cam1 not in positioned_cameras:
				# Position cam1 relative to cam2 (reference) - need to invert
				target_camera = cam1
				R = stereo_result['R'].T  # Invert rotation
				T = -stereo_result['R'].T @ stereo_result['T'].reshape(3, 1)  # Invert translation
			else:
				continue  # Skip if both cameras already positioned or neither is reference
			
			# Create 4x4 transform matrix
			transform = np.eye(4, dtype=np.float64)
			transform[:3, :3] = R
			transform[:3, 3] = T.flatten()
			
			# Convert to camera pose (inverse of the transform)
			camera_pose = np.linalg.inv(transform)
			
			self.world_poses[target_camera] = {
				'world_transform': camera_pose,
				'rvec': cv2.Rodrigues(camera_pose[:3, :3])[0],
				'tvec': camera_pose[:3, 3].reshape(3, 1),
				'method': 'stereo',
				'baseline_to_ref': stereo_result['baseline'],
				'stereo_error': stereo_result['rms_error']
			}
			
			positioned_cameras.add(target_camera)
			print(f"✓ {target_camera} positioned at {camera_pose[:3, 3]} (baseline: {stereo_result['baseline']:.3f}m)")
		
		# Step 6: Handle remaining cameras with simple calibration if needed
		remaining_cameras = set(self.cameras.keys()) - positioned_cameras
		if remaining_cameras:
			print(f"\nPositioning remaining cameras using simple calibration: {list(remaining_cameras)}")
			
			# Try to calibrate remaining cameras using simple method
			simple_poses = self.CalibrateMultiCameraExtrinsics("capture_3.tiff", save_results=False)
			
			if simple_poses:
				for camera_name in remaining_cameras:
					if camera_name in simple_poses:
						self.world_poses[camera_name] = simple_poses[camera_name]
						self.world_poses[camera_name]['method'] = 'simple_fallback'
						print(f"✓ {camera_name} positioned using simple calibration")
		
		# Step 7: Calculate camera relationships
		self.CalculateCameraRelationships()
		
		# Step 8: Save results
		if save_results:
			self.SaveMultiCameraCalibration()
			
			# Save stereo calibration details
			stereo_data = {
				'reference_camera': reference_camera,
				'stereo_pairs': {},
				'pair_analysis': pair_analysis,
				'shared_features_summary': {
					'num_images': len(shared_features['images']),
					'image_names': shared_features['image_names'][:20],  # Save first 20 names
					'detection_stats': shared_features['detection_stats']
				}
			}
			
			# Convert numpy arrays to lists for JSON serialization
			for pair_key, result in stereo_results.items():
				stereo_data['stereo_pairs'][pair_key] = {
					'camera1': result['camera1'],
					'camera2': result['camera2'],
					'rms_error': float(result['rms_error']),
					'R': result['R'].tolist(),
					'T': result['T'].tolist(),
					'baseline': float(result['baseline']),
					'num_images': int(result['num_images'])
				}
			
			with open("stereo_calibration_results.json", 'w') as f:
				json.dump(stereo_data, f, indent=4)
			
			print("Saved stereo calibration results to stereo_calibration_results.json")
		
		print(f"✓ Multi-camera stereo calibration complete: {len(self.world_poses)} cameras positioned")
		return self.world_poses

	def BundleAdjustment(self, max_iterations=100, convergence_threshold=1e-6):
		"""
		Perform bundle adjustment to refine camera poses and 3D points.
		
		Args:
			max_iterations: Maximum optimization iterations
			convergence_threshold: Convergence threshold for optimization
		
		Returns:
			dict: Optimized camera poses
		"""
		print("→ Bundle adjustment optimization")
		
		try:
			from scipy.optimize import least_squares
		except ImportError:
			print("Warning: scipy not available, skipping bundle adjustment")
			return self.world_poses
		
		if not self.world_poses or len(self.world_poses) < 2:
			print("Error: Need at least 2 calibrated cameras for bundle adjustment")
			return None
		
		# Collect all observations across cameras
		shared_features = self.CollectSharedFeatures()
		
		if len(shared_features['images']) < 3:
			print("Error: Need more images with shared features")
			return self.world_poses
		
		# Build optimization problem
		camera_names = list(self.world_poses.keys())
		
		# Initial parameters: [camera_poses, 3d_points]
		initial_params = []
		
		# Camera poses (6 DOF each: 3 rotation + 3 translation)
		camera_param_indices = {}
		param_idx = 0
		
		for i, camera_name in enumerate(camera_names):
			if i == 0:  # Fix first camera as reference
				camera_param_indices[camera_name] = None
			else:
				pose = self.world_poses[camera_name]['world_transform']
				rvec = cv2.Rodrigues(pose[:3, :3])[0].flatten()
				tvec = pose[:3, 3]
				initial_params.extend(rvec)
				initial_params.extend(tvec)
				camera_param_indices[camera_name] = (param_idx, param_idx + 6)
				param_idx += 6
		
		# 3D points (collect unique corner IDs)
		all_corner_ids = set()
		for image_data in shared_features['images']:
			for camera_data in image_data['cameras'].values():
				all_corner_ids.update(camera_data['ids'])
		
		corner_ids_list = sorted(list(all_corner_ids))
		obj_corners = self.ChArUco_board.getChessboardCorners()
		
		point_param_indices = {}
		for corner_id in corner_ids_list:
			if corner_id < len(obj_corners):
				point_3d = obj_corners[corner_id]
				initial_params.extend(point_3d)
				point_param_indices[corner_id] = (param_idx, param_idx + 3)
				param_idx += 3
		
		initial_params = np.array(initial_params)
		
		# Build observation data
		observations = []
		camera_indices = []
		point_indices = []
		
		for image_data in shared_features['images']:
			for camera_name, cam_data in image_data['cameras'].items():
				cam_idx = camera_names.index(camera_name)
				
				for i, corner_id in enumerate(cam_data['ids']):
					if corner_id in point_param_indices:
						point_idx = corner_ids_list.index(corner_id)
						observations.append(cam_data['corners'][i])
						camera_indices.append(cam_idx)
						point_indices.append(point_idx)
		
		observations = np.array(observations)
		camera_indices = np.array(camera_indices)
		point_indices = np.array(point_indices)
		
		print(f"Bundle adjustment: {len(observations)} observations, {len(camera_names)} cameras, {len(corner_ids_list)} 3D points")
		
		def residual_function(params):
			"""Compute reprojection residuals."""
			residuals = []
			
			# Extract camera poses
			current_poses = {}
			current_poses[camera_names[0]] = np.eye(4)  # Reference camera
			
			for i, camera_name in enumerate(camera_names[1:], 1):
				param_range = camera_param_indices[camera_name]
				if param_range is not None:
					rvec = params[param_range[0]:param_range[0]+3]
					tvec = params[param_range[0]+3:param_range[1]]
					
					R = cv2.Rodrigues(rvec)[0]
					pose = np.eye(4)
					pose[:3, :3] = R
					pose[:3, 3] = tvec
					current_poses[camera_name] = pose
			
			# Extract 3D points
			current_points = {}
			for corner_id in corner_ids_list:
				if corner_id in point_param_indices:
					param_range = point_param_indices[corner_id]
					current_points[corner_id] = params[param_range[0]:param_range[1]]
			
			# Compute reprojection errors
			for obs_idx, (camera_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
				camera_name = camera_names[camera_idx]
				corner_id = corner_ids_list[point_idx]
				
				if corner_id in current_points:
					# Project 3D point to camera
					point_3d = current_points[corner_id]
					pose = current_poses[camera_name]
					
					# Transform to camera coordinates
					point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]
					
					# Project to image
					K = self.cameras[camera_name]['K']
					dist = self.cameras[camera_name]['dist']
					
					if point_cam[2] > 0:  # Point in front of camera
						projected, _ = cv2.projectPoints(
							point_cam.reshape(1, 1, 3),
							np.zeros(3), np.zeros(3),
							K, dist
						)
						projected_2d = projected[0, 0]
						
						# Residual
						observed_2d = observations[obs_idx]
						residual = projected_2d - observed_2d
						residuals.extend(residual)
					else:
						residuals.extend([1000.0, 1000.0])  # Large error for invalid projection
			
			return np.array(residuals)
		
		# Run optimization
		print("Running bundle adjustment optimization...")
		try:
			result = least_squares(
				residual_function,
				initial_params,
				max_nfev=max_iterations * len(initial_params),
				ftol=convergence_threshold,
				verbose=1
			)
			
			if result.success:
				print(f"✓ Bundle adjustment converged in {result.nfev} iterations")
				print(f"Final cost: {result.cost:.6f}")
				
				# Extract optimized camera poses
				optimized_poses = {}
				optimized_poses[camera_names[0]] = np.eye(4)  # Reference camera
				
				for i, camera_name in enumerate(camera_names[1:], 1):
					param_range = camera_param_indices[camera_name]
					if param_range is not None:
						rvec = result.x[param_range[0]:param_range[0]+3]
						tvec = result.x[param_range[0]+3:param_range[1]]
						
						R = cv2.Rodrigues(rvec)[0]
						pose = np.eye(4)
						pose[:3, :3] = R
						pose[:3, 3] = tvec
						optimized_poses[camera_name] = pose
						
						# Update world poses
						self.world_poses[camera_name]['world_transform'] = pose
						self.world_poses[camera_name]['rvec'] = rvec.reshape(3, 1)
						self.world_poses[camera_name]['tvec'] = tvec.reshape(3, 1)
						self.world_poses[camera_name]['method'] = 'bundle_adjusted'
				
				print("✓ Camera poses updated with bundle adjustment results")
				return self.world_poses
			else:
				print("✗ Bundle adjustment failed to converge")
				return self.world_poses
				
		except Exception as e:
			print(f"Bundle adjustment error: {e}")
			return self.world_poses

	def AnalyzeImageQuality(self, max_images=50):
		"""
		Analyze image quality across all cameras to identify best images for calibration.
		
		Args:
			max_images: Maximum number of images to analyze
		
		Returns:
			dict: Analysis results with recommendations
		"""
		print("→ Analyzing image quality for calibration")
		
		if not self.cameras:
			self.LoadCameraIntrinsics()
		
		# Get common images
		all_images = set()
		for camera_data in self.cameras.values():
			folder = camera_data['folder']
			images = [f.name for f in folder.glob("capture_*.tiff")]
			if not all_images:
				all_images = set(images)
			else:
				all_images = all_images.intersection(set(images))
		
		image_names = sorted(list(all_images))[:max_images]
		print(f"Analyzing {len(image_names)} common images")
		
		image_analysis = []
		
		for img_idx, img_name in enumerate(image_names):
			if img_idx % 10 == 0:
				print(f"  Analyzing images {img_idx+1}-{min(img_idx+10, len(image_names))}...")
			
			image_data = {
				'name': img_name,
				'cameras': {},
				'total_cameras': 0,
				'total_corners': 0,
				'min_corners': float('inf'),
				'max_corners': 0,
				'corner_variance': 0,
				'quality_score': 0
			}
			
			corner_counts = []
			
			for camera_name, camera_data in self.cameras.items():
				image_path = camera_data['folder'] / img_name
				if not image_path.exists():
					continue
				
				img = cv2.imread(str(image_path))
				if img is None:
					continue
				
				# Detect ChArUco board
				res = self.FindCharucoBoard(img)
				
				if res is not None:
					char_corners = res['charucoCorners']
					char_ids = res['charucoIds']
					
					if char_corners is not None and char_ids is not None and len(char_corners) >= 4:
						corner_count = len(char_corners)
						corner_counts.append(corner_count)
						
						image_data['cameras'][camera_name] = {
							'corners': corner_count,
							'ids': char_ids.flatten().tolist()
						}
						
						image_data['total_cameras'] += 1
						image_data['total_corners'] += corner_count
						image_data['min_corners'] = min(image_data['min_corners'], corner_count)
						image_data['max_corners'] = max(image_data['max_corners'], corner_count)
			
			# Calculate quality metrics
			if corner_counts:
				image_data['corner_variance'] = np.var(corner_counts)
				avg_corners = np.mean(corner_counts)
				
				# Quality score based on:
				# - Number of cameras detecting the board (40%)
				# - Average corners per camera (30%)
				# - Consistency across cameras (low variance) (30%)
				camera_score = (image_data['total_cameras'] / len(self.cameras)) * 40
				corner_score = min(avg_corners / 35, 1.0) * 30  # 35 is max expected corners
				consistency_score = max(0, 30 - image_data['corner_variance'])
				
				image_data['quality_score'] = camera_score + corner_score + consistency_score
				image_data['avg_corners'] = avg_corners
			else:
				image_data['min_corners'] = 0
				image_data['avg_corners'] = 0
			
			image_analysis.append(image_data)
		
		# Sort by quality score
		image_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
		
		# Print analysis results
		print(f"\n📊 Image Quality Analysis Results:")
		print(f"{'Rank':<4} {'Image':<15} {'Cams':<5} {'Avg Corners':<11} {'Quality':<7} {'Status'}")
		print("-" * 60)
		
		excellent_images = []
		good_images = []
		fair_images = []
		
		for i, img_data in enumerate(image_analysis[:20]):  # Show top 20
			rank = i + 1
			name = img_data['name']
			cams = img_data['total_cameras']
			avg_corners = img_data.get('avg_corners', 0)
			quality = img_data['quality_score']
			
			if quality >= 80:
				status = "🟢 Excellent"
				excellent_images.append(name)
			elif quality >= 60:
				status = "🟡 Good"
				good_images.append(name)
			elif quality >= 40:
				status = "🟠 Fair"
				fair_images.append(name)
			else:
				status = "🔴 Poor"
			
			print(f"{rank:<4} {name:<15} {cams:<5} {avg_corners:<11.1f} {quality:<7.1f} {status}")
		
		# Recommendations
		print(f"\n💡 Recommendations:")
		print(f"• Excellent images ({len(excellent_images)}): Use for primary calibration")
		print(f"• Good images ({len(good_images)}): Use as additional data")
		print(f"• Fair images ({len(fair_images)}): Use only if needed")
		
		if excellent_images:
			print(f"\n🎯 Best images to use: {excellent_images[:10]}")
		elif good_images:
			print(f"\n🎯 Best available images: {good_images[:10]}")
		else:
			print(f"\n⚠️  No high-quality images found. Consider:")
			print(f"   • Better lighting conditions")
			print(f"   • Closer camera positioning")
			print(f"   • Ensuring board is visible from all cameras")
		
		return {
			'analysis': image_analysis,
			'excellent': excellent_images,
			'good': good_images,
			'fair': fair_images,
			'recommendations': excellent_images[:10] if excellent_images else good_images[:10]
		}

	def RunMultiCam(self):
		"""
		Run the multi-camera calibration process.
		"""
		print("→ Running multi-camera calibration")
		
		# Step 1: Load camera intrinsics
		cameras = self.LoadCameraIntrinsics()

		if not cameras:
			print("❌ No cameras found! Make sure intrinsics.json files exist in camera folders.")
			return

		print(f"✅ Loaded {len(cameras)} cameras: {list(cameras.keys())}")

		# Step 2: Choose calibration method
		print("\n2. Multi-camera calibration methods available:")
		print("   A) Simple board-based calibration (fast, less accurate)")
		print("   B) Stereo calibration with shared features (better accuracy)")
		print("   C) Stereo + Bundle adjustment (best accuracy, slower)")

		print("\n   Using stereo calibration with shared features...")
		world_poses = self.CalibrateMultiCameraStereo(
			reference_camera='Camera1',
			save_results=True
		)
  
		if not world_poses:
			print("❌ Stereo calibration failed!")
			return


		print(f"✅ Calibrated {len(world_poses)} cameras using {'stereo'} method")


		# Step 3: Print calibration summary
		print("\n3. Calibration Summary:")
		self.PrintCalibrationSummary()

		# Step 4: Validate calibration quality
		print("\n4. Validating calibration quality...")
		self.ValidateCalibration(test_image_name="capture_10.tiff")
  
  		#output extrinsics to table_cam_ext1 - table_cam_ext4
		for idx, each in enumerate(self.world_poses):

			self.ExtrinsicsToTable(self.world_poses[each]['rvec'], self.world_poses[each]['tvec'], op('table_cam_ext'+str(idx+1)))

	def DiagnoseCalibrationIssues(self, test_images=None):
		"""
		Diagnose calibration issues and suggest fixes.
		
		Args:
			test_images: List of test image names, or None to auto-select
		
		Returns:
			dict: Diagnostic results and recommendations
		"""
		print("🔍 Diagnosing calibration issues...")
		
		if not self.cameras:
			self.LoadCameraIntrinsics()
		
		if test_images is None:
			# Use a few different test images
			test_images = ["capture_3.tiff", "capture_10.tiff", "capture_50.tiff", "capture_100.tiff"]
		
		diagnostics = {
			'detection_issues': {},
			'calibration_issues': {},
			'recommendations': []
		}
		
		# Test detection across multiple images
		print("\n1. Testing board detection across cameras...")
		for camera_name, camera_data in self.cameras.items():
			detection_stats = {
				'total_tested': 0,
				'successful_detections': 0,
				'corner_counts': [],
				'failed_images': []
			}
			
			for test_img in test_images:
				image_path = camera_data['folder'] / test_img
				if not image_path.exists():
					continue
				
				img = cv2.imread(str(image_path))
				if img is None:
					continue
				
				detection_stats['total_tested'] += 1
				res = self.FindCharucoBoard(img)
				
				if res is not None and res['charucoCorners'] is not None:
					corner_count = len(res['charucoCorners'])
					if corner_count >= 4:
						detection_stats['successful_detections'] += 1
						detection_stats['corner_counts'].append(corner_count)
					else:
						detection_stats['failed_images'].append(test_img)
				else:
					detection_stats['failed_images'].append(test_img)
			
			detection_rate = (detection_stats['successful_detections'] / detection_stats['total_tested'] * 100) if detection_stats['total_tested'] > 0 else 0
			avg_corners = np.mean(detection_stats['corner_counts']) if detection_stats['corner_counts'] else 0
			
			diagnostics['detection_issues'][camera_name] = detection_stats
			
			status = "✅" if detection_rate >= 75 else "⚠️" if detection_rate >= 50 else "❌"
			print(f"  {status} {camera_name}: {detection_rate:.1f}% detection rate, {avg_corners:.1f} avg corners")
			
			if detection_rate < 50:
				diagnostics['recommendations'].append(f"Camera {camera_name}: Very low detection rate - check camera position and lighting")
			elif detection_rate < 75:
				diagnostics['recommendations'].append(f"Camera {camera_name}: Moderate detection rate - consider adjusting camera angle")
		
		# Test calibration quality if available
		if self.world_poses:
			print("\n2. Testing calibration quality...")
			validation_results = self.ValidateCalibration("capture_10.tiff")
			
			for camera_name, result in validation_results.items():
				if result['status'] == 'failed':
					diagnostics['recommendations'].append(f"Camera {camera_name}: Calibration validation failed - recalibration needed")
				elif result['status'] == 'high_error':
					diagnostics['recommendations'].append(f"Camera {camera_name}: Very high reprojection error - check intrinsic calibration")
				elif result['status'] == 'moderate_error':
					diagnostics['recommendations'].append(f"Camera {camera_name}: High reprojection error - consider using more/better images")
			
			diagnostics['calibration_issues'] = validation_results
		
		# Check camera coverage
		print("\n3. Analyzing camera coverage...")
		shared_features = self.CollectSharedFeatures(min_shared_corners=4)
		
		pair_coverage = {}
		camera_names = list(self.cameras.keys())
		
		for i, cam1 in enumerate(camera_names):
			for j, cam2 in enumerate(camera_names):
				if i < j:
					pair_key = f"{cam1}_{cam2}"
					shared_count = 0
					
					for image_data in shared_features['images']:
						if cam1 in image_data['cameras'] and cam2 in image_data['cameras']:
							shared_count += 1
					
					pair_coverage[pair_key] = shared_count
					status = "✅" if shared_count >= 10 else "⚠️" if shared_count >= 5 else "❌"
					print(f"  {status} {cam1} ↔ {cam2}: {shared_count} shared images")
					
					if shared_count < 5:
						diagnostics['recommendations'].append(f"Camera pair {cam1}-{cam2}: Very few shared views - adjust camera positions for better overlap")
		
		# Generate summary recommendations
		print("\n💡 Diagnostic Summary and Recommendations:")
		if not diagnostics['recommendations']:
			print("✅ No major issues detected!")
		else:
			for i, rec in enumerate(diagnostics['recommendations'], 1):
				print(f"{i}. {rec}")
		
		# Suggest next steps
		print("\n🎯 Suggested Next Steps:")
		
		# Check if simple calibration might work better
		poor_detection_cameras = [cam for cam, stats in diagnostics['detection_issues'].items() 
								 if (stats['successful_detections'] / stats['total_tested'] * 100) < 50]
		
		if len(poor_detection_cameras) >= 2:
			print("1. Try simple board-based calibration instead of stereo calibration")
			print("2. Improve lighting conditions and camera positioning")
			print("3. Use fewer, higher-quality images")
		elif self.world_poses and any(result.get('error', 0) > 10 for result in validation_results.values()):
			print("1. Check intrinsic calibration quality")
			print("2. Recalibrate with better images")
			print("3. Try bundle adjustment for refinement")
		else:
			print("1. Current calibration appears reasonable")
			print("2. Consider bundle adjustment for optimization")
		
		return diagnostics

	def TrySimpleCalibrationFallback(self, save_results=True):
		"""
		Try simple board-based calibration with automatic best image selection.
		
		Args:
			save_results: Whether to save results
		
		Returns:
			dict: Calibration results or None if failed
		"""
		print("🔄 Trying simple calibration fallback...")
		
		# First analyze image quality to find best reference image
		analysis = self.AnalyzeImageQuality(max_images=50)
		
		if not analysis['excellent'] and not analysis['good']:
			print("❌ No good quality images found for calibration")
			return None
		
		# Try the best images as reference
		candidate_images = analysis['excellent'][:5] if analysis['excellent'] else analysis['good'][:5]
		
		best_result = None
		best_error = float('inf')
		best_image = None
		
		for ref_image in candidate_images:
			print(f"\n🧪 Testing with reference image: {ref_image}")
			
			try:
				# Try calibration with this reference image
				temp_poses = self.CalibrateMultiCameraExtrinsics(
					reference_image_name=ref_image,
					save_results=False
				)
				
				if temp_poses and len(temp_poses) >= 2:
					# Validate this calibration
					self.world_poses = temp_poses  # Temporarily set for validation
					validation_results = self.ValidateCalibration(test_image_name=ref_image)
					
					# Calculate average error
					valid_errors = [result['error'] for result in validation_results.values() 
								   if result['status'] == 'success' and result['error'] < 1000]
					
					if valid_errors:
						avg_error = np.mean(valid_errors)
						print(f"  Average validation error: {avg_error:.3f} pixels")
						
						if avg_error < best_error:
							best_error = avg_error
							best_result = temp_poses.copy()
							best_image = ref_image
							print(f"  ✅ New best result!")
					else:
						print(f"  ❌ Validation failed")
				else:
					print(f"  ❌ Calibration failed")
					
			except Exception as e:
				print(f"  ❌ Error: {str(e)}")
		
		if best_result is not None:
			print(f"\n🎯 Best simple calibration result:")
			print(f"• Reference image: {best_image}")
			print(f"• Average error: {best_error:.3f} pixels")
			print(f"• Cameras calibrated: {len(best_result)}")
			
			# Set the best result
			self.world_poses = best_result
			
			# Calculate relationships
			self.CalculateCameraRelationships()
			
			if save_results:
				self.SaveMultiCameraCalibration()
			
			return best_result
		else:
			print("❌ All simple calibration attempts failed")
			return None
