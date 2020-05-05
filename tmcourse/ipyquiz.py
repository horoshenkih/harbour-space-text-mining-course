import ipywidgets as widgets
import typing as tp
from copy import deepcopy
from tabulate import tabulate


class Quiz:
    def __init__(
        self,
        description: tp.Union[str, widgets.Output],
        options: tp.List[tp.Any],
        answer: tp.Any
    ):
        assert answer in options, "answer is not in options"
        self._description = description
        self._options = options[:]
        self._answer = answer

    def __call__(self):
        if type(self._description) == str:
            description = widgets.Label(value=self._description)
        else:
            description = self._description
        choices = widgets.RadioButtons(
            options=self._options,
            disabled=False
        )
        output = widgets.HTML("<br>")

        def check_answer(*args, **kwargs):
            if choices.value == self._answer:
                output.value = "<font color='green'>'{}' is the correct answer</font>".format(choices.value)
            else:
                output.value = "<font color='red'>'{}' is an incorrect answer</font>".format(choices.value)

        check = widgets.Button(description="Check")
        check.on_click(check_answer)
        return widgets.VBox([description, choices, output, check])


class Function:
    def __init__(
            self,
            description: tp.Union[str, widgets.Output],
            etalon_solution: tp.Optional[tp.Callable],
            input_list=tp.Optional[tp.List[tp.List[tp.Any]]],
            input_output_list=tp.Optional[tp.List[tp.Tuple[tp.List[tp.Any], tp.Any]]],
            show_n_answers=1
    ):
        self._description = description
        self._show_n_answers = show_n_answers
        input_list = input_list or []
        input_output_list = input_output_list or []

        self._input_output_list = []
        if not etalon_solution:
            assert not input_list, "input_list is not allowed unless etalon_function is provided"
            self._input_output_list = deepcopy(input_output_list)
        else:
            seen_input_args = set()
            for input_args, output in input_output_list:
                input_args = tuple(input_args)
                if input_args in seen_input_args:
                    continue
                assert etalon_solution(
                    *input_args) == output, "etalon solution doesn't pass the test ({}) -> {}".format(input_args,
                                                                                                      output)
                seen_input_args.add(input_args)  # FIXME no nested structures are checked
                self._input_output_list.append((input_args, output))
            for input_args in input_list:
                input_args = tuple(input_args)
                if input_args in seen_input_args:
                    continue
                seen_input_args.add(input_args)  # FIXME no nested structures are checked
                self._input_output_list.append((input_args, etalon_solution(*input_args)))

        assert len(
            self._input_output_list) >= self._show_n_answers, "list of input-output pairs is too short ({} items) to show {} answers".format(
            len(self._input_output_list), self._show_n_answers)

    def __call__(self, solution: tp.Callable):
        if type(self._description) == str:
            description = widgets.HTML(value=str(self._description) + "<hr>")
        else:
            description = self._description

        sample_input_output = self._input_output_list[:self._show_n_answers]
        if len(sample_input_output):
            table = [(repr(i), repr(o)) for i, o in sample_input_output]
            sample = widgets.HTML("Sample input-output pairs:<br>" + tabulate(table, tablefmt='html',
                                                                              headers=["input", "output"]) + "<hr>")
        else:
            sample = widgets.HTML("")

        test_indicators = [o == solution(*i) for i, o in self._input_output_list]
        num_tests = len(test_indicators)
        num_correct_tests = sum(test_indicators)
        is_correct_solution = all(test_indicators)
        if is_correct_solution:
            verdict = widgets.HTML("<font color='green'>All {} tests passed</font>".format(num_tests))
        else:
            fail_message = "<font color='red'>{} / {} tests failed</font>".format(num_tests - num_correct_tests,
                                                                                  num_tests)
            table = [(repr(i), repr(solution(*i))) for i, o in sample_input_output]
            verdict = widgets.HTML("<br>".join([fail_message, "Answers on sample input-output pairs:",
                                                tabulate(table, tablefmt='html', headers=["input", "output"])]))
        return widgets.VBox([description, sample, verdict])
