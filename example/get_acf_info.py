import xpc

def get_acf_info(client):
    aircraft_icao = client.getDREF("sim/aircraft/view/acf_size_z")
    print(f"sim/aircraft/view/acf_size_z: {aircraft_icao}")

    aircraft_icao = client.getDREF("sim/aircraft/view/acf_size_x")
    print(f"sim/aircraft/view/acf_size_x: {aircraft_icao}")

    aircraft_desc = client.getDREF("sim/aircraft/view/acf_descrip")
    byte_data = bytes(int(x) for x in aircraft_desc if x != 0)
    description = byte_data.decode('ascii')
    print("acf_descrip:", description)

    aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    icao_code = byte_data.decode('ascii')
    print("ICAO code:", icao_code)
    # current_plane_length = client.getDREF('sim/aircraft/view/acf_length')
    # print(f"Plane length: {current_plane_length} ft")
    # current_plane_width = client.getDREF("sim/aircraft/parts/acf_wing_ft")
    # print(f"Plane width: {current_plane_width} ft")
    # current_plane_height = client.getDREF("sim/aircraft/parts/acf_htail_ft")
    # print(f"Plane height: {current_plane_height} ft")
    # # current_plane_mass = client.getDREF("sim/aircraft/weight/acf_m_empty")
    # print(f"Plane mass: {current_plane_mass} kg")
    # current_plane_fuel = client.getDREF("sim/aircraft/weight/acf_m_fuel_max")
    # print(f"Plane fuel: {current_plane_fuel} kg")
    # current_plane_fuel_density = client.getDREF("sim/aircraft/weight/acf_fuel_density")
    # print(f"Plane fuel density: {current_plane_fuel_density} kg/l")
    # current_plane_fuel_volume = current_plane_fuel / current_plane_fuel_density
    # print(f"Plane fuel volume: {current_plane_fuel_volume} l")
    # current_plane_fuel_volume_gal = current_plane_fuel_volume * 264.172
    # current_plane_fuel_volume_l = current_plane_fuel_volume * 1000
    # current_plane_fuel_volume_kg = current_plane_fuel_volume * 0.72
    # current_plane_fuel_volume_lb = current_plane_fuel_volume * 6.25
    # current_plane_fuel_volume_kg = current_plane_fuel_volume * 0.72

    
    
    
    
    
    
    # print(f"Plane fuel volume: {current_plane_fuel_volume} l")

with xpc.XPlaneConnect() as client:
    get_acf_info(client)
