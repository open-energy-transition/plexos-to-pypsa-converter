def check_constraints(network):
    print("Constraint Summary for PyPSA Network\n" + "-" * 40)

    # 1. Global Constraints
    if not network.global_constraints.empty:
        print("\nGlobal Constraints:")
        print(network.global_constraints.to_string())
    else:
        print("\nGlobal Constraints: None")

    # 2. Generator Constraints
    gen = network.generators
    if not gen.empty:
        print("\nGenerator Constraints:")
        print(
            gen[
                ["p_nom", "p_nom_max", "ramp_limit_up", "ramp_limit_down", "efficiency"]
            ].dropna(how="all")
        )
    else:
        print("\nGenerator Constraints: None")

    # 3. Storage Constraints
    su = network.storage_units
    if not su.empty:
        print("\nStorage Unit Constraints:")
        print(
            su[
                ["p_nom", "max_hours", "efficiency_store", "efficiency_dispatch"]
            ].dropna(how="all")
        )
    else:
        print("\nStorage Unit Constraints: None")

    # 4. Line Constraints
    lines = network.lines
    if not lines.empty:
        print("\nLine Constraints:")
        print(lines[["s_nom", "s_nom_max", "x", "r"]].dropna(how="all"))
    else:
        print("\nLine Constraints: None")

    # 5. Transformer Constraints
    trafos = network.transformers
    if not trafos.empty:
        print("\nTransformer Constraints:")
        print(trafos[["s_nom", "x", "r", "tap_ratio"]].dropna(how="all"))
    else:
        print("\nTransformer Constraints: None")

    print("\nConstraint summary complete.\n")
