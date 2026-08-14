LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

TARGET_TO_SLUG = {
    "ACL": "acl",
    "MCL": "mcl",
    "Medial Meniscus": "medial_meniscus",
    "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa",
    "Lateral OA": "lateral_oa",
    "PF OA": "pf_oa",
    "Effusion": "effusion",
    "Synovitis": "synovitis",
    "Baker's": "bakers",
    "Contusion": "contusion",
    "Fracture": "fracture",
}

TARGET_QUERIES = {
    "ACL": "ACL anterior cruciate ligament tear rupture partial tear sprain MRI diagnostic criteria",
    "MCL": "MCL medial collateral ligament tear sprain injury MRI diagnostic criteria",
    "Medial Meniscus": "medial meniscus tear root radial horizontal longitudinal complex MRI criteria",
    "Lateral Meniscus": "lateral meniscus tear root radial horizontal longitudinal complex MRI criteria",
    "Medial OA": "medial tibiofemoral osteoarthritis cartilage loss chondrosis osteophyte MRI criteria",
    "Lateral OA": "lateral tibiofemoral osteoarthritis cartilage loss chondrosis osteophyte MRI criteria",
    "PF OA": "patellofemoral osteoarthritis patellar trochlear cartilage chondrosis MRI criteria",
    "Effusion": "knee joint effusion MRI definition severity fluid diagnostic criteria",
    "Synovitis": "knee synovitis synovial thickening inflammation MRI diagnostic criteria",
    "Baker's": "Baker cyst popliteal cyst MRI diagnostic criteria",
    "Contusion": "knee contusion bone bruise muscle contusion marrow edema MRI diagnostic criteria",
    "Fracture": "knee fracture insufficiency subchondral osteochondral fracture MRI diagnostic criteria",
}
