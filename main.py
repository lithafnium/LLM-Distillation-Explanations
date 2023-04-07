from train_and_eval import Trainer 
import argparse

if __name__ == "__main__":
    t = Trainer
    parser = argparse.ArgumentParser(description="Lime Evaluation")

    parser.add_argument("--student", "-s", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--teacher", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--train_teacher", "-tt", action="store_true")
    parser.add_argument("--train_student", "-ts", action="store_true")
    parser.add_argument("--task", type=str, default="sst2")

    args = parser.parse_args() 

    t = Trainer(
        lr=5e-5, 
        batch_size=16,
        epochs=3,
        teacher_type=args.teacher,
        student_type=args.student,
        train_teacher=args.train_teacher,
        train_student=args.train_student,
        task=args.task
    )

    teacher, student = t.train_and_eval()