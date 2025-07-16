Templating System
================

MarrmotFlow uses a Jinja2-based templating system to generate MARRMOT model configuration files. This system provides flexibility in creating model instances and customizing parameters.

Overview
--------

The templating system in MarrmotFlow:

* Generates MATLAB code for MARRMOT models
* Uses Jinja2 templates for flexibility
* Supports multiple model structures
* Allows parameter customization
* Handles model-specific configurations

Template Structure
------------------

Default Template
~~~~~~~~~~~~~~~~

MarrmotFlow includes a default template for MARRMOT models:

.. code-block:: text

   src/marrmotflow/templates/marrmot_models.m.jinja

This template contains the structure for generating MATLAB model files compatible with the MARRMOT toolbox.

Template Variables
~~~~~~~~~~~~~~~~~~

The template system uses several variables that are populated during workflow execution:

.. code-block:: jinja2

   % Model: {{ model_name }}
   % Model number: {{ model_number }}
   % Catchment: {{ catchment_name }}
   % Generated: {{ timestamp }}

   function [theta] = {{ function_name }}()
       % Generated MARRMOT model configuration
       % Model parameters
       theta = [
           {% for param in parameters %}
           {{ param.value }}, % {{ param.name }} - {{ param.description }}
           {% endfor %}
       ];
   end

Using the Template System
--------------------------

Basic Usage
~~~~~~~~~~~

The templating system is used automatically during workflow execution:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow

   # Templates are used automatically
   workflow = MARRMOTWorkflow(
       name="TemplatedWorkflow",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       model_number=[7, 37]  # Templates will be generated for both models
   )

Direct Template Usage
~~~~~~~~~~~~~~~~~~~~~

You can also use the templating functions directly:

.. code-block:: python

   from marrmotflow.templating import render_models

   # Render models directly
   model_files = ["model_7.m", "model_37.m"]
   rendered_models = render_models(model_files)

   # Output is a dictionary with file names as keys and content as values
   for filename, content in rendered_models.items():
       print(f"Generated {filename}")
       print(content[:200] + "...")  # First 200 characters

Template Customization
----------------------

Custom Templates
~~~~~~~~~~~~~~~~

You can create custom templates for specific needs:

.. code-block:: python

   # Custom template path
   from marrmotflow.templating import render_models

   custom_template = "my_custom_template.m.jinja"
   rendered = render_models(
       model_files=["custom_model.m"],
       template_jinja_path=custom_template
   )

Template Context
~~~~~~~~~~~~~~~~

Understanding the context passed to templates:

.. code-block:: python

   # Template context includes:
   template_context = {
       "model_number": 7,
       "model_name": "HBV-96",
       "catchment_id": "basin_001",
       "catchment_name": "Example Basin",
       "parameters": [
           {"name": "TT", "value": 0.0, "description": "Temperature threshold"},
           {"name": "C0", "value": 3.0, "description": "Degree-day factor"},
           # ... more parameters
       ],
       "forcing_data": {
           "precipitation": "precip_data",
           "temperature": "temp_data",
           "pet": "pet_data"
       },
       "timestamp": "2024-01-01 12:00:00",
       "workflow_name": "MyWorkflow"
   }

Advanced Template Features
--------------------------

Conditional Logic
~~~~~~~~~~~~~~~~~

Templates can include conditional logic:

.. code-block:: jinja2

   % Model configuration for {{ model_name }}
   {% if model_number == 7 %}
   % HBV-96 specific configuration
   snow_routine = true;
   {% elif model_number == 37 %}
   % GR4J specific configuration
   snow_routine = false;
   {% endif %}

   % Parameters
   {% for param in parameters %}
   {% if param.active %}
   {{ param.name }} = {{ param.value }}; % {{ param.description }}
   {% endif %}
   {% endfor %}

Loops and Iterations
~~~~~~~~~~~~~~~~~~~~

Templates support loops for repetitive content:

.. code-block:: jinja2

   % Catchment data
   {% for catchment in catchments %}
   catchment_{{ loop.index }} = struct();
   catchment_{{ loop.index }}.id = '{{ catchment.id }}';
   catchment_{{ loop.index }}.name = '{{ catchment.name }}';
   catchment_{{ loop.index }}.area = {{ catchment.area }};
   {% endfor %}

Error Handling
~~~~~~~~~~~~~~

Templates include error handling capabilities:

.. code-block:: jinja2

   {% if not parameters %}
   {{ raise("No parameters defined for model " + model_number|string) }}
   {% endif %}

   {% for param in parameters %}
   {% if param.value is none %}
   {{ raise("Parameter " + param.name + " has no value") }}
   {% endif %}
   {% endfor %}

Template Development
--------------------

Creating Custom Templates
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom template:

1. **Create a new .jinja file**:

.. code-block:: jinja2

   % Custom MARRMOT Model Template
   % Model: {{ model_name }} ({{ model_number }})
   % Workflow: {{ workflow_name }}
   % Generated: {{ timestamp }}

   function [output] = run_{{ model_name|lower|replace('-', '_') }}(forcing_data)
       % Custom model implementation
       
       % Model parameters
       {% for param in parameters %}
       {{ param.name }} = {{ param.value }}; % {{ param.description }}
       {% endfor %}
       
       % Model logic here
       output = struct();
       output.discharge = []; % Model output
   end

2. **Register the template** (if using custom template system):

.. code-block:: python

   from marrmotflow.templating import render_models

   custom_models = render_models(
       model_files=["custom_model.m"],
       template_jinja_path="path/to/custom_template.m.jinja"
   )

Template Best Practices
-----------------------

Organization
~~~~~~~~~~~~

Keep templates organized and well-documented:

.. code-block:: jinja2

   {# 
   Template: MARRMOT Model Generator
   Purpose: Generate MATLAB code for MARRMOT models
   Author: Your Name
   Date: {{ timestamp }}
   #}

   % Generated MARRMOT Model
   % Do not edit this file directly - it is auto-generated

Variable Naming
~~~~~~~~~~~~~~~

Use clear, descriptive variable names:

.. code-block:: jinja2

   % Clear parameter definitions
   {% for param in model_parameters %}
   {{ param.matlab_name }} = {{ param.calibrated_value }}; % {{ param.physical_meaning }} [{{ param.units }}]
   {% endfor %}

Comments and Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

Include comprehensive comments:

.. code-block:: jinja2

   % ================================================================
   % MARRMOT Model: {{ model_name }}
   % ================================================================
   % Description: {{ model_description }}
   % Reference: {{ model_reference }}
   % 
   % Parameters:
   {% for param in parameters %}
   %   {{ param.name }}: {{ param.description }} [{{ param.units }}]
   {% endfor %}
   % ================================================================

Template Testing
----------------

Validating Templates
~~~~~~~~~~~~~~~~~~~

Test your templates with different inputs:

.. code-block:: python

   def test_template(template_path, test_contexts):
       """Test template with various contexts."""
       from jinja2 import Environment, FileSystemLoader
       
       env = Environment(loader=FileSystemLoader('.'))
       template = env.get_template(template_path)
       
       for i, context in enumerate(test_contexts):
           try:
               result = template.render(context)
               print(f"Test {i+1}: SUCCESS")
           except Exception as e:
               print(f"Test {i+1}: FAILED - {e}")

   # Test contexts
   test_contexts = [
       {"model_number": 7, "parameters": []},
       {"model_number": 37, "parameters": [{"name": "X1", "value": 100}]},
   ]
   
   test_template("marrmot_models.m.jinja", test_contexts)

Template Debugging
~~~~~~~~~~~~~~~~~~

Debug template issues:

.. code-block:: python

   # Enable template debugging
   from jinja2 import Environment, FileSystemLoader, DebugUndefined

   env = Environment(
       loader=FileSystemLoader('.'),
       undefined=DebugUndefined  # Shows undefined variables
   )

Integration with Workflow
-------------------------

Automatic Template Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Templates are processed automatically during workflow execution:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       name="AutoTemplating",
       # ... other parameters
   )

   # Templates are generated and used internally
   # Generated files can be accessed after workflow execution

Manual Template Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate templates manually for inspection:

.. code-block:: python

   from marrmotflow.templating import render_models

   # Generate templates for specific models
   model_codes = render_models(["model_7.m", "model_37.m"])

   # Save to files for inspection
   for filename, code in model_codes.items():
       with open(f"generated_{filename}", 'w') as f:
           f.write(code)
       print(f"Saved generated_{filename}")

Future Extensions
-----------------

The templating system is designed to be extensible:

* Support for additional model formats
* Custom parameter optimization templates
* Integration with other modeling frameworks
* Template sharing and version control

Best Practices Summary
----------------------

1. **Use descriptive names** for all template variables
2. **Include comprehensive comments** in generated code
3. **Test templates thoroughly** with various inputs
4. **Handle errors gracefully** with appropriate error messages
5. **Keep templates modular** and easy to maintain
6. **Document template purpose** and usage
7. **Version control templates** along with code changes
