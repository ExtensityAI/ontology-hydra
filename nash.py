import numpy as np
from symai import Symbol
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import umap
import matplotlib.pyplot as plt
import io
from PIL import Image

#TODO:
    # better way to score questions (biggest bottleneck)
    # better prompting
    # better persona creation
    # better question generation

log_dir = f"runs/nash_game_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir)

model = SentenceTransformer('all-MiniLM-L6-v2')

personas = [
    "A domain expert in machine learning.",
    "A curious researcher asking fundamental questions.",
    "A critic evaluating the validity of the questions.",
    "An AI system generating exploratory questions."
]

def evaluate_question(persona, question):
    """Have a persona evaluate a question and provide feedback."""
    prompt = f"""Evaluate this question: "{question}"
    Rate it from 1-10 and provide brief feedback.
    Return your response in format: score|feedback"""

    response = Symbol(f"You are {persona}.").query(prompt)
    try:
        score, feedback = response.value.split('|')
        return float(score), feedback.strip()
    except:
        return 5.0, "Neutral feedback"

def cross_persona_review(questions, persona_source, other_personas):
    """Have other personas review questions from one persona."""
    reviews = []
    for question in questions:
        question_reviews = []
        for reviewer in other_personas:
            if reviewer != persona_source:
                score, feedback = evaluate_question(reviewer, question)
                question_reviews.append({
                    'reviewer': reviewer,
                    'score': score,
                    'feedback': feedback
                })
        reviews.append({
            'question': question,
            'reviews': question_reviews,
            'avg_score': np.mean([r['score'] for r in question_reviews])
        })
    return reviews

def refine_questions(persona, original_questions, feedback):
    """Refine questions based on collective feedback."""
    feedback_summary = "\n".join([
        f"Question: {review['question']}\n"
        f"Average Score: {review['avg_score']}\n"
        f"Feedback: {'; '.join([r['feedback'] for r in review['reviews']])}"
        for review in feedback
    ])

    prompt = f"""These were your original questions:
    {original_questions}

    Here's the feedback received:
    {feedback_summary}

    Generate improved versions of your questions based on this feedback.
    Return exactly {len(original_questions)} questions, separated by newlines."""

    refined = Symbol(f"You are {persona}.").query(prompt)
    return refined.value.split('\n')

def generate_cqs(persona, num_questions=1):
    """Simulate LLM-based CQ generation for a given persona."""
    prompt = f"Generate {num_questions} key questions for knowledge graph construction. Return the questions separated by a newline."
    res = Symbol(persona).query(prompt)
    return res.value.split("\n")

def consensus_similarity(cq_sets):
    """Compute pairwise cosine similarity to determine consensus level."""
    flattened_cqs = [cq for sublist in cq_sets for cq in sublist]
    embeddings = model.encode(flattened_cqs)
    similarity_matrix = cosine_similarity(embeddings)
    return np.mean(similarity_matrix)

def calculate_persona_similarities(cq_sets):
    """Calculate similarities between personas' questions."""
    persona_similarities = {}
    for i, persona1 in enumerate(personas):
        for j, persona2 in enumerate(personas[i+1:], i+1):
            if i != j:
                emb1 = model.encode(cq_sets[i])
                emb2 = model.encode(cq_sets[j])
                sim = np.mean(cosine_similarity(emb1, emb2))
                persona_similarities[f"{persona1[:10]}-{persona2[:10]}"] = sim
    return persona_similarities

def plot_umap_embeddings(embeddings, labels, iteration):
    """Create UMAP visualization of embeddings."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])

    for i, txt in enumerate(labels):
        plt.annotate(txt[:30] + "...", (umap_embeddings[i, 0], umap_embeddings[i, 1]))

    plt.title(f'UMAP Projection of Questions (Iteration {iteration})')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return image

def has_converged(previous_cqs, current_cqs, threshold=0.9):
    """Check if consensus has been reached."""
    if not previous_cqs:
        return False
    prev_sim = consensus_similarity(previous_cqs)
    curr_sim = consensus_similarity(current_cqs)
    return curr_sim - prev_sim < 0.01 and curr_sim > threshold

def iterative_consensus_game(max_rounds=5, threshold=0.9):
    """Run iterative consensus game with feedback mechanism."""
    iteration = 0
    previous_cqs = []

    while iteration < max_rounds:
        print(f"\nIteration {iteration + 1}:")
        current_cqs = []
        refined_cqs = []

        for i, persona in enumerate(personas):
            prompt = f"You are {persona} working on knowledge graph construction."
            cqs = generate_cqs(prompt)
            current_cqs.append(cqs)
            print(f"\n{persona} initial questions: {cqs}")

            other_personas = personas[:i] + personas[i+1:]
            reviews = cross_persona_review(cqs, persona, other_personas)

            refined = refine_questions(persona, cqs, reviews)
            refined_cqs.append(refined)
            print(f"{persona} refined questions: {refined}")

            for j, review in enumerate(reviews):
                writer.add_scalar(
                    f'Question_Scores/{persona[:10]}/Q{j}',
                    review['avg_score'],
                    iteration
                )

        current_consensus = consensus_similarity(refined_cqs)
        writer.add_scalar('Consensus/Overall_Similarity', current_consensus, iteration)

        persona_similarities = calculate_persona_similarities(refined_cqs)
        for pair, sim in persona_similarities.items():
            writer.add_scalar(f'Similarities/{pair}', sim, iteration)

        flattened_refined_cqs = [cq for sublist in refined_cqs for cq in sublist]
        embeddings = model.encode(flattened_refined_cqs)
        umap_image = plot_umap_embeddings(embeddings, flattened_refined_cqs, iteration)
        writer.add_image('UMAP_visualization', np.array(umap_image).transpose(2, 0, 1), iteration)

        writer.add_embedding(
            torch.tensor(embeddings),
            metadata=flattened_refined_cqs,
            global_step=iteration,
            tag=f'question_embeddings/iteration_{iteration}'
        )

        if has_converged(previous_cqs, refined_cqs, threshold):
            print("\nConsensus Reached! Final set of refined CQs:")
            writer.add_text('Convergence', 'Consensus reached', iteration)
            return refined_cqs

        previous_cqs = refined_cqs
        iteration += 1
        plt.close('all')

    print("\nMax iterations reached. Returning last set of refined CQs.")
    writer.add_text('Convergence', 'Max iterations reached', iteration)
    return refined_cqs

final_cqs = iterative_consensus_game(max_rounds=5)

final_cqs_flat = [cq for sublist in final_cqs for cq in sublist]
writer.add_text('Final Questions', '\n'.join(final_cqs_flat))

print("\nFinal Consensus Questions:")
for i, cq in enumerate(final_cqs_flat):
    print(f"{i+1}. {cq}")

writer.close()
