function Motor3_Down(port_num, PROTOCOL_VERSION)
ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
ADDR_GOAL_POSITION       = 30;
ADDR_PRESENT_POSITION    = 36;
ADDR_MOVING_SPEED = 32;
ADDR_TORQUE_LIMIT = 34;

M = 4;
M3 = 4;


  write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 50);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1200);

    pause(2);
    
     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1024);
     
       write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 80);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 15);
    
    pause(0.5)
    
     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);

end

