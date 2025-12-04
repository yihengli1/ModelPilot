from .services import generate_plan_from_gpt
from rest_framework import status
from rest_framework.response import Response


def training_pipeline(system_context, prompt, dataset):

    # Hyperparameter initialization 1 Call

    llm_result = training_initialization(system_context, prompt, dataset)

    parsing_initialization(llm_result)

    # feature selection

    # Split Model, PyTorch training

    # Based on results 2 call

    # iterate over range of models/hyperparams/

    # best validation error

    # send back result

    pass


def training_initialization(system_context, prompt, dataset):
    try:
        llm_result = generate_plan_from_gpt(
            system_context=system_context,
            prompt=prompt,
            dataset=dataset,
        )
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except EnvironmentError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except ImportError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception:
        return Response(
            {"error": "Failed to generate plan from LLM."},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return llm_result


# Output Example
# {
#   "problem_type": "...",
#   "target_column": "...",
#   "recommended_models": [
#     {
#       "model": "...",
#       "reasoning": "...",
#       "initial_hyperparameters": {
#           "param1": ...,
#           "param2": ...,
#           "etc": ...
#        }
#     }
#   ],
#   "data_split": {
#     "method": "...",
#     "train_val_test": [ ..., ..., ... ],
#     "stratify": "...",
#     "grouping_column": "..."
#   }
# }


def parsing_initialization(llm_result):
    problem_type = llm_result.get("problem_type")
    target_column = llm_result.get("target_column")
    data_split = llm_result.get("data_split", {})
    recommended_models = llm_result.get("recommended_models", [])

    model_plans = []
    for model in recommended_models:
        model_name = model.get("model")
        reasoning = model.get("reasoning")
        model_key = model_name.lower().replace(" ", "_")
        hyperparams = model.get("initial_hyperparameters")

        model_plans.append(
            {
                "model": model_key,
                "hyperparameters": hyperparams,
                "reasoning": reasoning
            }
        )

    return {
        problem_type,
        target_column,
        data_split,
        model_plans
    }
