function Motor3_Up(port_num, PROTOCOL_VERSION)

M = 4;
ADDR_MOVING_SPEED = 32;
ADDR_PRESENT_POSITION = 36;

top_pos = 3306;

dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_PRESENT_POSITION);

write2ByteTxRx(port_num, PROTOCOL_VERSION, M, 34, 400);

if abs(dxl_present_position-top_pos) > 20
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_MOVING_SPEED, 25);
    while abs(dxl_present_position-top_pos) > 20
        dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_PRESENT_POSITION);
    end
end

dxl_present_position = read2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_PRESENT_POSITION)
write2ByteTxRx(port_num, PROTOCOL_VERSION, M, ADDR_MOVING_SPEED, 0);

end

