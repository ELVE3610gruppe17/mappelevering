import numpy as np
import cv2 as cv
import math
import chess
import chess.engine
import socket

### Global variables

oslomet_cam = True

CAM_WIDTH = 1920    #1280    #1440
CAM_HEIGHT = 1080    #720   #600
marker_size = 24
mirrored_camera = False
aruco_type = "DICT_4X4_100"
ref_aruco = 69
camera = 1
checker_size = 34.7
circle_radius = 38
scale_percent = 60
#X_avg = 346.1
#Y_avg = 847.1

X_avg = 351.7
Y_avg = 836.2 
#836.2
continue_game = True

###
client_ip = "192.168.12.62"
client_port = 2222

stockfish_path = r"D:\ELVE3610 - Robotikk\Ferdig_prosjekt\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe"

intrinsic_camera = np.array(((1.41963350e+03, 0.00000000e+00, 9.29823698e+02), (0.00000000e+00, 1.41912081e+03, 5.57347229e+02), (0.00000000e+00, 0.00000000e+00, 1.00000000e+00) ))
distortion = np.array((0.02685585, -0.13528379, -0.00116169, -0.00088772,  0.09404331)) #webcam

"""intrinsic_camera = np.array(((1.41963350e+03, 0.00000000e+00, 9.29823698e+02), (0.00000000e+00, 1.41912081e+03, 5.57347229e+02), (0.00000000e+00, 0.00000000e+00, 1.00000000e+00) ))
distortion = np.array((0.02685585, -0.13528379, -0.00116169, -0.00088772,  0.09404331))""" #uni


ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

starting_positions = (
    [ 1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16],
    [0 ,0 ,0 ,0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32]
)
### Functions  

def aruco_display_and_pos(corners, ids, rejected, image):
    first = True
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            cv.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            if first:
                marker_pos = np.array([ [markerID, cX , cY] ])
                first = False
            else:
                marker_pos = np.append(marker_pos, [[markerID, cX , cY]], axis=0)
            cv.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
            #print("[Inference] ArUco marker ID: {}".format(markerID))
            #print(f"Marker pos: {marker_pos}")
        
    if first == True:
        marker_pos = np.array([ [0,0,0] ])
			
    return image, marker_pos

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv.solvePnP(marker_points, corners[i], mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.array(rvecs), np.array(tvecs), np.array(trash)

def pose_estimation(frame, frame_copy, aruco_dict_type, arucoDetect,matrix_coefficients, distortion_coefficients):
    found_ref = False # in case it doesn't find the reference frame
    
    checkerboard = np.zeros((8,8,2))       ##Array for storing center point of each checker
    
    
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
    #parameters = cv.aruco.DetectorParameters()
    
    

    #corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters,
    #    cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    
    corners, ids, rejected_img_points = arucoDetect.detectMarkers(frame_copy)

    if len(corners) > 0:
        
        
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = my_estimatePoseSingleMarkers(corners[i], marker_size, matrix_coefficients,
                                                                       distortion_coefficients)
            if ids[i] == ref_aruco:
                #print(f"Found reference marker {ids[i]}")
                found_ref = True
                cv.aruco.drawDetectedMarkers(frame_copy, corners)
                
                ###TEST
                #x, y, z = 0.04, 0., 0.
                #mat_a = np.array( [ [1., 0., 0., x], [0., 1., 0., y], [0., 0., 1., z], [0., 0., 0., 1.] ] )
                #new_vec = np.vstack([tvec[0], [1.]])
                #temp = mat_a.dot(new_vec)
                #temp = np.array([np.delete(temp,(3), axis=0)])
                
            
                #new_tvec = np.copy(tvec)
                #print(f"copy: {new_tvec[0][0]}")
                #new_tvec[0][0] += 0.0000005
                #print(f"copy: {new_tvec[0][0]}")

                # Project the new point onto the image
                
                #cv.circle(image, circle_center, circle_radius, (0, 255, 0), 2)
                #print(tuple(new_image_point[0].astype(int).ravel()))
                k = 1
                
                for j in range(0,8): 
                    for i in range (0, 8):
                        x_shift = checker_size*i  # one checker along the positive x-axis
                        y_shift = checker_size*(j+1)
                        marker_coords = np.array([x_shift, y_shift, 0], dtype=np.float32).reshape(-1, 1, 3)

                        cam_coords, _ = cv.projectPoints(marker_coords, rvec, tvec, matrix_coefficients,distortion_coefficients)

                        checkerboard[7-j][i] = cam_coords[0].astype(int).ravel()
                        k = k+1

                        dot_color = (0, 0, 255)  # Red circle
                        #dot_size = 25#+(j+1)*2
                        cv.circle(frame, tuple(cam_coords[0].astype(int).ravel()), circle_radius, dot_color, 5)
                        cv.circle(frame, tuple(cam_coords[0].astype(int).ravel()), 5, dot_color, -1)
                #image_with_dot = cv.circle(frame, tuple(checkerboard[0][3].astype(int).ravel()), dot_size, dot_color, 2)
                #print(f"Hey: {tuple(checkerboard[7][7])}")
                ###TEST
                
                cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)        
                #print(f"r vec: {rvec}\n")
    if found_ref == False:
        rvec = np.array([ [0. ,0. ,0. ] ])
        tvec = np.array([ [0. ,0., 0.]])
    return frame, rvec, tvec, checkerboard

def interpret_chessboard(frame, checkerboard, pieces):
    board_id = np.zeros((8,8),dtype=int)
    found_markers = []
    #print(board[0][0])
    for i in range(7,-1, -1):
        for j in range(0,8):
            for element in pieces:
                distance = np.sqrt((element[1] - checkerboard[i][j][0])**2 + (element[2] - checkerboard[i][j][1])**2)
                if distance < 25 and element[0] not in found_markers:
                    found_markers.append(element[0])
                    board_id[i][j] = element[0]
                    dot_color = (0, 255, 0)  # Green circle
                    cv.circle(frame, tuple(checkerboard[i][j].astype(int).ravel()), circle_radius, dot_color, 5)
                elif distance < 25:
                    dot_color = (255, 0, 0)  # Blue circle
                    cv.circle(frame, tuple(checkerboard[i][j].astype(int).ravel()), circle_radius, dot_color, 5)
                    print("Double positive - ignored")
    
    return board_id

def fen_representation(chess_array):
    fen = ""
    count = False
    empty_slots = 0

    pieces_dict = {
        "r" : [1, 8],
        "n" : [2, 7],
        "b" : [3, 6],
        "q" : 4,
        "k" : 5,
        "p" : range(9,17),
        "P" : range(17,25),
        "R" : [25, 32],
        "N" : [26, 31],
        "B" : [27, 30],
        "Q" : 28,
        "K" : 29
    }
    
    for i in range (0,8):
        for j in range(0,8):
            #print(get_key_by_value(pieces_dict, chess_array[i][j]))
            if chess_array[i][j] == 0:
                count = True
                empty_slots += 1
            elif chess_array[i][j] != 0 and count == True:
                fen = fen + str(empty_slots) + get_key_by_value(pieces_dict, chess_array[i][j])
                empty_slots = 0
                count = False
            else:
                fen = fen + get_key_by_value(pieces_dict, chess_array[i][j])
        if count == True:
            fen = fen + str(empty_slots)
            empty_slots = 0
            count = False
        if i < 7:
            fen = fen + "/"
    
    return fen

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if isinstance(val, list):
            if value in val:
                return key
        elif isinstance(val, range):
            if value in val:
                return key
        elif val == value:
            return key
    return 'E'

#############CODE BEGINS HERE

def aruco_scan():
    proceed = False
    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv.aruco.DetectorParameters()
    
    #Variables to be changed if it's not picking up 
    arucoParams.cornerRefinementMinAccuracy = 0.05   #default: 0.01
    arucoParams.cornerRefinementWinSize = 20
    
    arucoDetect = cv.aruco.ArucoDetector(arucoDict, arucoParams)

    
    while proceed == False:

        
        cap = cv.VideoCapture(camera, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT , CAM_HEIGHT)
        
        succes, image = cap.read()
        #print("Image taken!")
        image_copy = image.copy()
        cap.release()
        
        if mirrored_camera:
            image = cv.flip(image, 1)
            image_copy = cv.flip(image_copy, 1)
            
            brightness = -150 #-125
            contrast = 100
            image = np.int16(image)
            image = image * (contrast/127+1) - contrast + brightness
            image = np.clip(image, 0, 255)
            image = np.uint8(image)
            image_copy = np.int16(image_copy)
            image_copy = image_copy * (contrast/127+1) - contrast + brightness
            image_copy = np.clip(image_copy, 0, 255)
            image_copy = np.uint8(image_copy)
            
        corners, ids, rejected = arucoDetect.detectMarkers(image)
        detected_markers, aruco_pos = aruco_display_and_pos(corners, ids, rejected, image)
        output, rot_vec, tra_vec, checker_array = pose_estimation(image, image_copy, ARUCO_DICT[aruco_type], arucoDetect, intrinsic_camera, distortion)

        board_array = interpret_chessboard(image, checker_array, aruco_pos)

        fen_string = fen_representation(board_array)

        #test
        """rot_mat, jac = cv.Rodrigues(rot_vec)
        #print("MarkerID - X - Y")
        #print(f"{aruco_pos}\n")
        theta_z = np.arctan2(rot_mat[1][0], rot_mat[0][0])
        theta_z = theta_z * -1
        #print(f"Theta Z: {theta_z*180/3.14} grader\n")
        #print(f"{tra_vec[0]}")
        #print(f"{tra_vec[0][0][0]}")
        
        #print(f"{tuple(checker_array[0][7])}")
        
        
        
        #print(f"fen: {fen_string}")
        
        ##bullshit##
        circle_x = 754
        circle_y = 542
        checker_length = 50
        n = -2
        m = 2
        new_x = checker_length * n * np.cos(theta_z) - checker_length * m * np.sin(theta_z)
        new_y = checker_length * n * np.sin(theta_z) + checker_length * m * np.cos(theta_z)
        circle_center = (circle_x+int(new_x), circle_y+int(new_y))
        circle_radius = 25
        cv.circle(image, circle_center, circle_radius, (0, 255, 0), 2)
        """## /end test
        #cv.imshow('Image',image)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
      
        # resize image
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        cv.imshow('Image',resized)
        
        while True:
            print("Looking good? [y/n]")
            word = cv.waitKey(0)
            if word == ord('y'):
                print("Proceeding")
                proceed = True
                break
            elif word == ord('n'):
                print("Scanning again")
                break
            else:
                print("Invalid input")
                continue
        
        cv.imwrite('images/checkerboard.png', image)
        
        #print("image saved!")
    cv.destroyAllWindows()
    return fen_string, board_array, aruco_pos, checker_array

def board_setup(client):
    setup_array = ['s']
    move_from = [0, 0]
    move_to = [0, 0]
    junk, board_ids, aruco_cam_pos, checkers = aruco_scan()
    for piece in aruco_cam_pos:
        if piece[0] in range(1,33):
            move_from[0] = piece[1]
            move_from[1] = piece[2]

        for i, row in enumerate(starting_positions):
            for j, col in enumerate(row):
                if starting_positions[i][j] == piece[0]:
                    move_to[0] = checkers[i][j][0]
                    move_to[1] = checkers[i][j][1]
                    break
        setup_array.append(move_to)
        setup_array.append(move_from)
        
    instructions_array = pixel_to_mm(setup_array, CAM_WIDTH ,CAM_HEIGHT, X_avg, Y_avg)
    
    send_setup(client, instructions_array)
    
    print("Setup is finished.")    
    return

### Chess engine

def get_board(fen): 
    Incomplete_FEN = fen
    Whose_turn = "b"
    Castling_rights = "kq"
    En_passant_rights = "-"
    Half_moves_wo_pawn = 0
    Complete_turns = 1

    Complete_FEN = f"{Incomplete_FEN} {Whose_turn} {Castling_rights} {En_passant_rights} {Half_moves_wo_pawn} {Complete_turns}"
    #print(f"The complete FEN-code: {Complete_FEN}")

    board = chess.Board(Complete_FEN)
    return board

def get_computer_move(board): # For stockfish path, download stockfish 16, unpack, add "r" before the string, copy path to the folder, and add "\stockfish-windows-x86-64-modern.exe"
    stockfish_path = r"D:\ELVE3610 - Robotikk\Ferdig_prosjekt\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe"
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=1.0))
        computer_move_algebraic = board.san(result.move)
        board.push(result.move)
        print(board)
        UCI_Algebraic = f"{result.move} {computer_move_algebraic}"
        #print(UCI_Algebraic)
        return UCI_Algebraic
    
def move_instructions_pixel(move, chessboard_ids, aruco_pixel_pos, checker_pos):
    special = 'o'
    move_elim = [0,0]
    move_from = [0,0]
    move_to = [0,0]
    cast_from = [0,0]
    cast_to = [0,0]
    
    row_from = 8-int(move[1])
    col_from = ord(move[0]) - ord('a')


    row_to = 8-int(move[3])
    col_to = ord(move[2]) - ord('a')
    
    #long castling
    if 'O-O-O' in move:
        special = '-'
        cast_id = chessboard_ids[0][0]
        for piece in aruco_pixel_pos:
            if piece[0] == cast_id:
                cast_from[0] = piece[1]
                cast_from[1] = piece[2]
        cast_to[0] = checker_pos[0][3][0]
        cast_to[1] = checker_pos[0][3][1]
    #short castling
    elif 'O-O' in move:
        special = '-'
        cast_id = chessboard_ids[0][7]
        for piece in aruco_pixel_pos:
            if piece[0] == cast_id:
                cast_from[0] = piece[1]
                cast_from[1] = piece[2]
        cast_to[0] = checker_pos[0][5][0]
        cast_to[1] = checker_pos[0][5][1]
    #checks if a piece is eliminated
    elif 'x' in move:
        special = 'x'
        elim_id = chessboard_ids[row_to][col_to]
        for piece in aruco_pixel_pos: 
            if piece[0] == elim_id:
                move_elim[0] = piece[1]
                move_elim[1] = piece[2]
                
    #finds pick-up point for move token
    move_id = chessboard_ids[row_from][col_from]
    for piece in aruco_pixel_pos:
        if piece[0] == move_id:
            move_from[0] = piece[1]
            move_from[1] = piece[2]
        
    #finds drop-off point for move token
    move_to[0] = checker_pos[row_to][col_to][0]
    move_to[1] = checker_pos[row_to][col_to][1]

    
    instr = [special, move_from, move_to, move_elim, cast_from, cast_to]
    
    print(f"HELLO: {instr}")
    return instr

def change_parameters():
    global CAM_WIDTH 
    global CAM_HEIGHT 
    global mirrored_camera 
    global circle_radius
    global scale_percent
    global intrinsic_camera
    global distortion
    if oslomet_cam == True:
        CAM_WIDTH = 1440 
        CAM_HEIGHT = 600
        mirrored_camera = True
        circle_radius = math.ceil( checker_size * math.sqrt( (X_avg/CAM_HEIGHT)**2 + (Y_avg/CAM_WIDTH)**2 ))  # 28
        scale_percent = 100
        intrinsic_camera = np.array(((1.41963350e+03, 0.00000000e+00, 9.29823698e+02), (0.00000000e+00, 1.41912081e+03, 5.57347229e+02), (0.00000000e+00, 0.00000000e+00, 1.00000000e+00) ))
        distortion = np.array((0.02685585, -0.13528379, -0.00116169, -0.00088772,  0.09404331))
    return

##Socket

def socket_client(host_ip,port): #Takes ip and port, establishes connection, and returns "client" for future use in socket connection definitions
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect((host_ip, port)) 
    return client

def send_message(client, special_case,
                 x_from, y_from, x_to, y_to,
                 x_taken,y_taken,
                x_from_rook, y_from_rook, x_to_rook, y_to_rook): #Sends moves to robot
    
    client.send(bytes(special_case, "utf-8"))
    print(client.recv(1024))

    if special_case != "q":
        client.send(bytes(x_to, "utf-8"))
        print(client.recv(1024))
        client.send(bytes(y_to, "utf-8"))
        print(client.recv(1024))
        client.send(bytes(x_from, "utf-8"))
        print(client.recv(1024))
        client.send(bytes(y_from, "utf-8"))
        print(client.recv(1024))
        print("To and from coordinates sent!")
        
        if special_case == "x":
            print(client.recv(1024))
            client.send(bytes(x_taken, "utf-8"))
            print(client.recv(1024))
            client.send(bytes(y_taken, "utf-8"))
            print(client.recv(1024))
            print("Taken coordinates sent!")

        elif special_case == "-":
            print(client.recv(1024))
            client.send(bytes(x_to_rook, "utf-8"))
            print(client.recv(1024))
            client.send(bytes(y_to_rook, "utf-8"))
            print(client.recv(1024))
            client.send(bytes(x_from_rook, "utf-8"))
            print(client.recv(1024))
            client.send(bytes(y_from_rook, "utf-8"))
            print(client.recv(1024))
            print("Rook coordinates sent!")
        
        elif special_case =="q":
            client.send(bytes(special_case, "utf-8"))
            print("special_case is 'q', and robot is terminated.")
            
    print("Waiting for confirmation from robot.")
    print(client.recv(1024))
    
def send_setup(client, array):
    special_case = array[0]
    test_var = int((len(array)-1)/2)
    print(type(test_var))
    
    print(test_var)
    for i in range(test_var):
        client.send(bytes(special_case, "utf-8"))
        print(client.recv(1024))
        
        client.send(bytes(str(array[(2*i+1)][0]), "utf-8"))
        print(client.recv(1024))
        client.send(bytes(str(array[(2*i+1)][1]), "utf-8"))
        print(client.recv(1024))
        client.send(bytes(str(array[(2*i+2)][0]), "utf-8"))
        print(client.recv(1024))
        client.send(bytes(str(array[(2*i+2)][1]), "utf-8"))
        print(client.recv(1024))
        
        
        
    client.send(bytes("ferdig", "utf-8"))
    print(client.recv(1024))
    
    return

def pixel_to_mm(coordinate_array,
                px_x_max,px_y_max,
                x_avg, y_avg):
    
    px_relation_x = x_avg/px_y_max
    px_relation_y = y_avg/px_x_max
    instructions_mm = [coordinate_array[0]]

    for i in range(len(coordinate_array)-1):
        instructions_mm.append([str(round(((CAM_HEIGHT-coordinate_array[i+1][1]) * px_relation_x+284),2)) ,str(round(((CAM_WIDTH-coordinate_array[i+1][0])*px_relation_y-419),2))])
    return instructions_mm

### program begins here

def game_loop():
    global continue_game
    player_won = False
    print("Welcome to the Chess Robot! :)")
    change_parameters()
    print("Trying to connect to robot...")
    client = socket_client(client_ip, client_port)
    print(client.recv(1024))

    """while True:
        word = input("Do you want the robot to set up the board? [y/n]\n")
        if word.lower() == 'n':
            break
        elif word.lower() == 'y':
            board_setup(client)
        else:
            print("Invalid input.")"""


    while continue_game:

        fen, board_ids, aruco_cam_pos, checkers   = aruco_scan()
        
        board = get_board(fen)
        
        move = get_computer_move(board)
        
        print(f"trekk:\n {move}")
        
        instructions_pixel = move_instructions_pixel(move, board_ids, aruco_cam_pos, checkers)
        
        #instructions_pixel = ['o', [0,0], [600,0], [0,0], [0,0], [0,0] ]
        
        instructions_mm = pixel_to_mm(instructions_pixel, CAM_WIDTH ,CAM_HEIGHT, X_avg, Y_avg)
        
        #print(f"instructions_pixel:\n {instructions_pixel}")
        
        #print(f"instructions_mm:\n {instructions_mm}")
        print(instructions_mm[1][0])
        send_message(client,instructions_mm[0], 
                     instructions_mm[1][0], instructions_mm[1][1], instructions_mm[2][0], instructions_mm[2][1],
                     instructions_mm[3][0],instructions_mm[3][1],
                     instructions_mm[4][0],instructions_mm[4][1],instructions_mm[5][0],instructions_mm[5][1])
        
        if '#' in move:
            player_won = False
            continue_game = False
        
        #continue_game = False
        print
    outcome = "won!" if player_won else "lost."
    print(f"Game over. You {outcome}!")
    return

game_loop()
    

