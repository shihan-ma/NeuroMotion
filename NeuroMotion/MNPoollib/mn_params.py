# Range of MU depth and MU angle within each muscle
DEPTH = {
    'ECRB': [0.0130, 0.0220],
    'ECRL': [0.0085, 0.0153],
    'PL': [0.0071, 0.0114],
    'FCU_u': [0.0084, 0.0168],
    'FCU_h': [0.0074, 0.0165],
    'ECU': [0.0092, 0.0168],
    'EDI': [0.0079, 0.0171],
    'FDSI': [0.0169, 0.0231],
    'FCU': [0.0074, 0.0168]
}

ANGLE = {
    'ECRB': [0.4946, 0.6632],
    'ECRL': [0.4607, 0.5109],
    'PL': [0.0540, 0.0956],
    'FCU_u': [0.7878, 0.8658],
    'FCU_h': [0.9897, 1.0],
    'ECU': [0.7194, 0.7779],
    'EDI': [0.5637, 0.6826],
    'FDSI': [0.1471, 0.2264],
    'FCU': [0.7878, 1.0]
}

MS_AREA = {
    'ECRB': 60.100,
    'ECRL': 70.715,
    'PL': 35.335,
    'FCU_u': 78.925,
    'FCU_h': 83.305,
    'ECU': 47.720,
    'EDI': 60.000,
    'FDSI': 25.335,
    'FCU': 162.23
}  # mm^2

NUM_MUS = {
    'ECRB': 186,
    'ECRL': 204,
    'PL': 164,
    'FCU_u': 205,
    'FCU_h': 217,
    'ECU': 180,
    'EDI': 186,
    'FDSI': 158,
    'FCU': 422
}

# Settings below match iEMG package (git@github.com:smetanadvorak/iemg_simulator.git)
mn_default_settings = {
    'rr': 50,
    'rm': 0.75,
    'rp': 100,
    'pfr1': 40,
    'pfrd': 10,
    'mfr1': 10,
    'mfrd': 5,
    'gain': 30,     # 0.3 per % MVC
    'c_ipi': 0.1,
    'frs1': 50,
    'frsd': 20,
}