from django import forms

class LPForm(forms.Form):
    METHOD_CHOICES = [
        ('graphical', 'Graphical Method'),
        ('simplex', 'Simplex Method'),
        ('transportation', 'Transportation Problem'),
    ]
    method = forms.ChoiceField(choices=METHOD_CHOICES, label="Select Method")
    objective = forms.CharField(label='Objective Function Coefficients (comma separated)', required=False)
    constraints = forms.CharField(label='Constraints Coefficients (comma separated per row)', required=False)
    rhs_values = forms.CharField(label='Constraint RHS Values (comma separated)', required=False)
    opt_type = forms.ChoiceField(choices=[('minimize', 'Minimize'), ('maximize', 'Maximize')], label='Optimization Type', required=False)

class TransportationForm(forms.Form):
    supply = forms.CharField(label='Supply (comma-separated)', required=True)
    demand = forms.CharField(label='Demand (comma-separated)', required=True)
    costs = forms.CharField(label='Cost Matrix (rows separated by semicolons, values by commas)', required=True)
    method = forms.ChoiceField(
        label='Select Method',
        choices=[
            ('northwest', 'Northwest Corner Method'),
            ('least_cost', 'Least Cost Method'),
            ('vam', "Vogel's Approximation Method (VAM)"),
        ],
        required=True,
    )
