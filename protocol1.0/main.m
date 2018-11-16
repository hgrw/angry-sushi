clc;
% clear all;

lib_name = '';

if strcmp(computer, 'PCWIN')
  lib_name = 'dxl_x86_c';
elseif strcmp(computer, 'PCWIN64')
  lib_name = 'dxl_x64_c';
elseif strcmp(computer, 'GLNX86')
  lib_name = 'libdxl_x86_c';
elseif strcmp(computer, 'GLNXA64')
  lib_name = 'libdxl_x64_c';
elseif strcmp(computer, 'MACI64')
  lib_name = 'libdxl_mac_c';
end

% Load Libraries
if ~libisloaded(lib_name)
    [notfound, warnings] = loadlibrary(lib_name, 'dynamixel_sdk.h', 'addheader', 'port_handler.h', 'addheader', 'packet_handler.h');
end

% Control table address
ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
ADDR_GOAL_POSITION       = 30;
ADDR_PRESENT_POSITION    = 36;
ADDR_MOVING_SPEED = 32;
ADDR_TORQUE_LIMIT = 34;

% Protocol version
PROTOCOL_VERSION            = 1.0;          % See which protocol version is used in the Dynamixel




% Default setting
M1                      = 3;            % Dynamixel ID: 1
M2 = 2;
M3 = 4;
BAUDRATE                    = 1000000;
DEVICENAME                  = 'COM5';       % Check which port is being used on your controller
                                            % ex) Windows: 'COM1'   Linux: '/dev/ttyUSB0' Mac: '/dev/tty.usbserial-*'
                                   
                                            
 % MOTOR SETUP
 
M1_SPEED = 20;
M1_TORQUE = 90;

M2_SPEED = 30;
M2_TORQUE = 70;

M3_SPEED = 10;
M3_TORQUE = 50;
                                            
TORQUE_ENABLE               = 2;            % Value for enabling the torque
TORQUE_DISABLE              = 0;            % Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 100;          % Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 4000;         % and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 10; 

COMM_SUCCESS                = 0;            % Communication Success result value
COMM_TX_FAIL                = -1001;        % Communication Tx Failed

% Initialize PortHandler Structs
% Set the port path
% Get methods and members of PortHandlerLinux or PortHandlerWindows
port_num = portHandler(DEVICENAME);

% Initialize PacketHandler Structs
packetHandler();

index = 1;
dxl_comm_result = COMM_TX_FAIL;             % Communication result
dxl_goal_position = [DXL_MINIMUM_POSITION_VALUE DXL_MAXIMUM_POSITION_VALUE];         % Goal position

dxl_error = 0;                              % Dynamixel error
dxl_present_position = 0;                   % Present position


% Open port
if (openPort(port_num))
    fprintf('Succeeded to open the port!\n');
else
    unloadlibrary(lib_name);
    fprintf('Failed to open the port!\n');
    input('Press any key to terminate...\n');
    return;
end


% Set port baudrate
if (setBaudRate(port_num, BAUDRATE))
    fprintf('Succeeded to change the baudrate!\n');
else
    unloadlibrary(lib_name);
    fprintf('Failed to change the baudrate!\n');
    input('Press any key to terminate...\n');
    return;
end

write1ByteTxRx(port_num, PROTOCOL_VERSION, M1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE);
write2ByteTxRx(port_num, PROTOCOL_VERSION, M1, ADDR_MOVING_SPEED, M1_SPEED);
write2ByteTxRx(port_num, PROTOCOL_VERSION, M1, ADDR_TORQUE_LIMIT, M1_TORQUE);

write1ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE);
write2ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_MOVING_SPEED, M2_SPEED);
write2ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_TORQUE_LIMIT, M2_TORQUE);

% 
% P_X = [0.5 0.5 0.48 0.48 0.5 0.5 0.48 0.48 0.47 0.46 0.45 0.44 0.43];
% P_Y = [-0.04 -0.06 -0.06 -0.04 0.02 0 0.02 0 0 0 0 0 0 0];

yOffset = 0.2;
xOffset = 0.235;

% [py, px] = getPath();

%pathY = -[py]/1000 + yOffset;
%pathX = [px]/1000 + xOffset;

pathY = -[281   251   221   190   160   129    99]/1000 + yOffset;
pathX = [130   147   164   180   197   214   230]/1000 + xOffset;

% Px = 0.2;
% Py = 0;
%PlaceDot(port_num, PROTOCOL_VERSION);

%Motor3_Down(port_num, PROTOCOL_VERSION)

% MoveTo(0.6, 0, port_num, PROTOCOL_VERSION);

%PlaceDot(port_num, PROTOCOL_VERSION);

for i = 1:length(pathX)
    Px = pathX(i);
    Py = pathY(i);
    MoveTo(Px,Py,port_num, PROTOCOL_VERSION);
    PlaceDot(port_num, PROTOCOL_VERSION);
end

% P_X = [0.2081    0.2094    0.2157    0.2185    0.2281];
% P_Y = [0.2091    0.2182    0.2192    0.2246    0.2343];

% DrawPath(port_num, PROTOCOL_VERSION, P_X, P_Y);
% MoveTo(0.4,0.1,port_num, PROTOCOL_VERSION); 
% PlaceDot(port_num, PROTOCOL_VERSION);
% MoveTo(0.4,-0.1,port_num, PROTOCOL_VERSION); 
% PlaceDot(port_num, PROTOCOL_VERSION);

% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 900);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1070);
% pause(1);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1024);

% Motor1_Move(port_num, PROTOCOL_VERSION, 150);
% 
% Motor2_Move(port_num,PROTOCOL_VERSION, 150); 

%write2ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_GOAL_POSITION, 519);

write1ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_ENABLE, TORQUE_ENABLE);
%write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, M3_SPEED);
write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, M3_TORQUE);
%write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_GOAL_POSITION, 1000);

% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 10);
% pause(1);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);

% write1ByteTxRx(port_num, PROTOCOL_VERSION, M3, 6, 1);
% write1ByteTxRx(port_num, PROTOCOL_VERSION, M3, 8, 1);
% 
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_GOAL_POSITION, 4000);
dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_PRESENT_POSITION)

% MoveTo(0.4,0.1,port_num, PROTOCOL_VERSION);
% 
% DrawDots2(port_num, PROTOCOL_VERSION);

% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1050);
% pause(1);
% Motor2_Move(port_num,PROTOCOL_VERSION, 120);
% pause(3);

% pause(1);
% Motor1_Move(port_num, PROTOCOL_VERSION, 150);

% Motor3_Up(port_num, PROTOCOL_VERSION);
% pause(1);
% Motor2_Move(port_num,PROTOCOL_VERSION, 150);

%     dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_PRESENT_POSITION)
%     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
%     pause(0.5);
%     dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_PRESENT_POSITION)
%     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);
%     dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_PRESENT_POSITION)

% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1050);
% pause(1);
% % pause(1);
% % 
% % pause(4);
% % write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1050);
% % pause(1);
% % write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1024);
% % pause(0.5);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 700);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
% pause(1);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);

% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
% pause(0.5);
% write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);

% DrawDots(port_num, PROTOCOL_VERSION);

% MoveTo(0.5,-0.1,port_num, PROTOCOL_VERSION);

%         write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 50);
%         write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1084);
%         pause(1);
%            write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
%     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
%       pause(0.2);
%       write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);
      
closePort(port_num);



