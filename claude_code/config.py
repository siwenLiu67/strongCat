class Config:
    num_jobs = 1
    num_machines = 2
    num_distributors = 1
    min_operations = 2
    max_operations = 2
    min_machines_per_op = 1
    max_machines_per_op = num_machines
    min_processing_time = 1
    max_processing_time = 1
    earliest_delivery_time = 1
    latest_delivery_time = 2
    min_delivery_requirements = 1
    max_delivery_requirements = 1
    min_load_ratio = 1


    max_time_steps = 10
    arrival_probability = 0
    arrival_batch_size = 2