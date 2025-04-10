import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import TransportationForm

def solve_graphical(c, A, b):
    """
    Solves a linear programming problem using the graphical method.
    """
    media_dir = settings.MEDIA_ROOT
    if not os.path.exists(media_dir):
        os.makedirs(media_dir)

    x_vals = np.linspace(0, 50, 400)
    plt.figure(figsize=(8, 6))

    # Plot constraints
    for i, (a, rhs) in enumerate(zip(A, b)):
        if a[1] != 0:
            y_vals = (rhs - a[0] * x_vals) / a[1]
            plt.plot(x_vals, y_vals, label=f"Constraint {i+1}")
        else:
            plt.axvline(rhs / a[0], label=f"Constraint {i+1}", color="red")

    # Solve the linear programming problem
    solution = linprog(c, A_ub=A, b_ub=b, method="highs")
    optimal_point = None
    optimal_value = None  # <-- Add this to ensure 4 return values

    if solution.success:
        optimal_point = (round(solution.x[0], 2), round(solution.x[1], 2))
        optimal_value = round(-solution.fun, 2) if c[0] < 0 else round(solution.fun, 2)
        plt.scatter(*optimal_point, color="red", marker="o", s=100, label="Optimal Solution")

    # Plot settings
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend()
    plt.title("Graphical Method - Feasible Region & Optimal Solution")
    plt.grid()

    # Save the plot
    img_filename = "graphical_solution.png"
    img_path = os.path.join(media_dir, img_filename)
    plt.savefig(img_path)
    plt.close()

    # Ensure four values are returned
    return solution, solution.message, optimal_point, f"{settings.MEDIA_URL}{img_filename}"

def solve_simplex(c, A, b):
    """
    Solves a linear programming problem using the Simplex method.
    """
    solution = linprog(c, A_ub=A, b_ub=b, method="highs")
    optimal_point = None
    if solution.success:
        optimal_point = [round(x, 2) for x in solution.x]

    return solution, solution.message, optimal_point

def graphical_view(request):
    """
    Handles the Graphical Method view.
    """
    if request.method == "POST":
        obj_coeffs = list(map(float, request.POST.get("objective").split(',')))
        constraints_coeffs = [
            list(map(float, row.split(','))) for row in request.POST.get("constraints").split("\n")
        ]
        rhs_values = list(map(float, request.POST.get("rhs_values").split(',')))
        optimization_type = request.POST.get("optimization").lower()

        maximize = optimization_type == "maximize"
        if maximize:
            obj_coeffs = [-c for c in obj_coeffs]

        solution, status, optimal_point, image_url = solve_graphical(obj_coeffs, constraints_coeffs, rhs_values)

        optimal_value = round(-solution.fun, 2) if solution.success and maximize else round(solution.fun, 2)

        return render(request, "solve_lp.html", {
            "method_name": "Graphical",
            "status": status,
            "optimal_value": optimal_value,
            "decision_variables": optimal_point if solution.success else "No feasible solution",
            "image_path": image_url  # Correctly pass the image path
        })
    
    return render(request, "solve_lp.html", {"method_name": "Graphical"})

def simplex_view(request):
    """
    Handles the Simplex Method view.
    """
    if request.method == "POST":
        obj_coeffs = list(map(float, request.POST.get("objective").split(',')))
        constraints_coeffs = [
            list(map(float, row.split(','))) for row in request.POST.get("constraints").split("\n")
        ]
        rhs_values = list(map(float, request.POST.get("rhs_values").split(',')))
        optimization_type = request.POST.get("optimization").lower()

        maximize = optimization_type == "maximize"
        if maximize:
            obj_coeffs = [-c for c in obj_coeffs]

        solution, status, optimal_point = solve_simplex(obj_coeffs, constraints_coeffs, rhs_values)

        optimal_value = round(-solution.fun, 2) if solution.success and maximize else round(solution.fun, 2)

        return render(request, "solve_lp.html", {
            "method_name": "Simplex",
            "status": status,
            "optimal_value": optimal_value,
            "decision_variables": optimal_point if solution.success else "No feasible solution",
            "image_path": None
        })
    
    return render(request, "solve_lp.html", {"method_name": "Simplex"})

def transportation_view(request):
    """
    Handles the Transportation Problem view.
    """
    if request.method == 'POST':
        form = TransportationForm(request.POST)
        if form.is_valid():
            supply = list(map(int, form.cleaned_data['supply'].split(',')))
            demand = list(map(int, form.cleaned_data['demand'].split(',')))
            costs = [list(map(int, row.split(','))) for row in form.cleaned_data['costs'].split(';')]
            method = form.cleaned_data['method']

            if method == 'northwest':
                allocation = northwest_corner_method(supply, demand, costs)
                total_cost = calculate_total_cost(allocation, costs)
            elif method == 'least_cost':
                allocation, total_cost = least_cost_method(supply, demand, costs)
            elif method == 'vam':
                allocation, total_cost = vogels_approximation_method(supply, demand, costs)

            return render(request, 'results.html', {'allocation': allocation, 'total_cost': total_cost})
    else:
        form = TransportationForm()

    return render(request, 'transportation_form.html', {'form': form})

def northwest_corner_method(supply, demand, costs):
    """
    Solves the Transportation Problem using the Northwest Corner Method.
    """
    m = len(supply)
    n = len(demand)
    allocation = [[0 for _ in range(n)] for _ in range(m)]
    i, j = 0, 0

    while i < m and j < n:
        quantity = min(supply[i], demand[j])
        allocation[i][j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity

        if supply[i] == 0:
            i += 1
        else:
            j += 1

    return allocation

def least_cost_method(supply, demand, costs):
    """
    Solves the Transportation Problem using the Least Cost Method.
    """
    m = len(supply)
    n = len(demand)
    allocation = [[0 for _ in range(n)] for _ in range(m)]
    total_cost = 0

    while True:
        min_cost = float('inf')
        min_i, min_j = -1, -1

        for i in range(m):
            for j in range(n):
                if supply[i] > 0 and demand[j] > 0 and costs[i][j] < min_cost:
                    min_cost = costs[i][j]
                    min_i, min_j = i, j

        if min_i == -1 or min_j == -1:
            break

        quantity = min(supply[min_i], demand[min_j])
        allocation[min_i][min_j] = quantity
        supply[min_i] -= quantity
        demand[min_j] -= quantity
        total_cost += quantity * min_cost

    return allocation, total_cost

import sys

def vogels_approximation_method(supply, demand, costs):
    """
    Solves the Transportation Problem using Vogel's Approximation Method (VAM).
    """
    m = len(supply)
    n = len(demand)
    allocation = [[0 for _ in range(n)] for _ in range(m)]
    total_cost = 0

    # Convert supply and demand to mutable lists
    supply = supply[:]
    demand = demand[:]

    iteration = 0  # Debugging: Count iterations to prevent infinite loop
    while sum(supply) > 0 and sum(demand) > 0:
        iteration += 1
        if iteration > 100:  # Prevent infinite loop
            print("Infinite loop detected!")
            break

        print(f"Iteration {iteration}: Supply: {supply}, Demand: {demand}")  # Debugging

        row_penalties = []
        col_penalties = []

        # Calculate row penalties
        for i in range(m):
            valid_costs = sorted([costs[i][j] for j in range(n) if demand[j] > 0])
            if len(valid_costs) > 1:
                row_penalties.append(valid_costs[1] - valid_costs[0])
            elif valid_costs:
                row_penalties.append(valid_costs[0])
            else:
                row_penalties.append(sys.maxsize)  # Large value for unused rows

        # Calculate column penalties
        for j in range(n):
            valid_costs = sorted([costs[i][j] for i in range(m) if supply[i] > 0])
            if len(valid_costs) > 1:
                col_penalties.append(valid_costs[1] - valid_costs[0])
            elif valid_costs:
                col_penalties.append(valid_costs[0])
            else:
                col_penalties.append(sys.maxsize)  # Large value for unused columns

        # Find the maximum penalty
        max_row_penalty = max(row_penalties, default=0)
        max_col_penalty = max(col_penalties, default=0)

        # Select row or column
        if max_row_penalty >= max_col_penalty:
            i = row_penalties.index(max_row_penalty)
            valid_cols = [j for j in range(n) if demand[j] > 0]
            if not valid_cols:
                print("Error: No valid columns left.")
                break
            j = min(valid_cols, key=lambda x: costs[i][x])
        else:
            j = col_penalties.index(max_col_penalty)
            valid_rows = [i for i in range(m) if supply[i] > 0]
            if not valid_rows:
                print("Error: No valid rows left.")
                break
            i = min(valid_rows, key=lambda x: costs[x][j])

        # Allocate goods
        quantity = min(supply[i], demand[j])
        allocation[i][j] = quantity
        total_cost += quantity * costs[i][j]

        # Debugging output
        print(f"Allocating {quantity} at cost {costs[i][j]} for cell ({i}, {j})")

        # Update supply and demand
        supply[i] -= quantity
        demand[j] -= quantity

    print("Final Allocation Matrix:")
    for row in allocation:
        print(row)

    return allocation, total_cost



def calculate_total_cost(allocation, costs):
    """
    Calculates the total cost of the Transportation Problem solution.
    """
    total_cost = 0
    for i in range(len(allocation)):
        for j in range(len(allocation[0])):
            total_cost += allocation[i][j] * costs[i][j]
    return total_cost