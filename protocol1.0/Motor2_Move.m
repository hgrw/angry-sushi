function Motor2_Move(port_num, PROTOCOL_VERSION, angle)

M = 2;
ADDR_GOAL_POSITION = 30;

goal_pos = 1023 - angle*1023/300;

write2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_GOAL_POSITION, goal_pos);

end

