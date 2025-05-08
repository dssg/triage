The `Auditioner` class is the main entry point for the Audition module. Users pass its constructor a database connection, information about the model groups to be evaluated, and a specification for a filter to prune the worst-performing models.

Other methods allow users to define more complex selection rules, list selected models, or plot results from the selection process.

::: triage.component.audition
    options:
        show_root_toc_entry: false
        group_by_category: true
        show_category_heading: true
        show_if_no_docstring: true
        