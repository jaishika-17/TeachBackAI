"""
ai.py -- TeachBack AI's knowledge base and NLP evaluation engine.

This file is intentionally independent of the web framework and the
database: it's pure Python (stdlib only, no pip installs required) so it
can be tested, swapped, or replaced with a real LLM-backed implementation
without touching backend.py at all. backend.py imports only the functions
in the "Public API" section at the bottom.

Pipeline (mirrors the product build plan, section 6.5):
  Preprocess -> Concept/Claim Extraction -> Semantic Comparison ->
  Evaluation -> Gap Classification -> Adaptive Question -> Repair ->
  Verification.
"""
import re
from difflib import SequenceMatcher

# ===========================================================================
# 1. KNOWLEDGE BASE
# ===========================================================================
# Keyed by slug. Each concept carries the structured "ConceptKnowledge"
# TeachBack AI evaluates explanations against: key points (facts the
# student must mention, or reasoning the student must walk through), and
# misconceptions (incorrect claims the evaluator watches for).
#
# key_point fields:
#   label       short id used to track this point across attempts
#   claim       the idea in plain English
#   keywords    matcher terms/synonyms (see keyword_hits() below)
#   type        "fact" (must be mentioned) | "reasoning" (must be explained, not just named)
#   importance  1 (minor) - 3 (critical) -- weights every score
#   challenge   the adaptive question asked when this point is the top gap
#   repair      {explanation, example, analogy, mini_question, mini_answer}

CONCEPTS = {
    "binary-search": {
        "name": "Binary Search",
        "subject": "Algorithms",
        "difficulty": "beginner",
        "description": "Finding a target value in a sorted collection by repeatedly halving the search space.",
        "learning_objective": (
            "Explain what Binary Search is, why the array must be sorted, how the search space "
            "shrinks each step, and why that makes the algorithm O(log n)."
        ),
        "prerequisites": ["arrays", "time-complexity"],
        "key_points": [
            {
                "label": "sorted_data",
                "claim": "The array must be sorted before binary search can be used.",
                "keywords": ["sorted", "sorted array", "ordered", "in order", "ascending", "sorted list"],
                "type": "fact", "importance": 3,
                "challenge": "What would go wrong if you ran binary search on an unsorted array?",
                "repair": {
                    "explanation": "Binary search decides which half to discard by comparing the middle value to the target. That comparison is only meaningful if everything to one side is guaranteed smaller and everything to the other side is guaranteed larger -- which only holds when the data is sorted.",
                    "example": "Searching for 7 in [4, 9, 1, 7, 3]: the middle element is 1, smaller than 7, but the array isn't ordered, so 7 could be on either side. Binary search has no way to safely discard a half.",
                    "analogy": "It's like looking up a word in a dictionary that's been shuffled -- flipping to the middle tells you nothing about which half your word is in.",
                    "mini_question": "If a list is [5, 1, 9, 2], can binary search reliably find 9?",
                    "mini_answer": "No -- the list isn't sorted, so comparing against the middle element doesn't tell you which half to search next.",
                },
            },
            {
                "label": "middle_element",
                "claim": "Each step compares the target to the middle element of the current range.",
                "keywords": ["middle element", "midpoint", "middle value", "compare", "middle index", "checks the middle"],
                "type": "fact", "importance": 2,
                "challenge": "Why does the algorithm specifically check the middle element instead of, say, the first one?",
                "repair": {
                    "explanation": "Checking the middle is what lets the algorithm discard half the remaining elements in one comparison -- checking the first element would only rule out one element at a time, which is what linear search does.",
                    "example": "In [1,3,5,7,9,11,13], the middle is 7. If the target is 11, everything at or below 7 is discarded in a single step.",
                    "analogy": "It's like a guessing game where you're told 'higher' or 'lower' after every guess -- guessing the middle of the remaining range eliminates the most possibilities per guess.",
                    "mini_question": "In a sorted list of 15 elements, how many elements are eliminated by checking the middle element once?",
                    "mini_answer": "About 7 elements on one side are eliminated immediately.",
                },
            },
            {
                "label": "search_space_reduction",
                "claim": "Based on the comparison, one half of the remaining search space is discarded.",
                "keywords": ["search space", "half", "discard", "eliminate", "narrows down", "reduces the range", "remaining half"],
                "type": "fact", "importance": 2,
                "challenge": "After one comparison, exactly what happens to the part of the array that can't contain the answer?",
                "repair": {
                    "explanation": "Because the array is sorted, if the target is greater than the middle element, the entire lower half (including the middle) can be safely thrown away -- it's guaranteed not to contain the target.",
                    "example": "Searching for 20 where the middle is 12: since 20 > 12, the entire lower half is discarded in that step.",
                    "analogy": "It's like elimination rounds in a tournament bracket -- once a half loses the comparison, it's out for good.",
                    "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "log_n_reasoning",
                "claim": "Repeatedly halving the search space produces O(log n) time complexity.",
                "keywords": ["log n", "o(log n)", "logarithmic", "halving", "halve", "repeated halving", "keeps halving", "divides in half each time"],
                "type": "reasoning", "importance": 3,
                "challenge": "Why does repeatedly reducing the search space by half result in O(log n) time?",
                "repair": {
                    "explanation": "Each comparison cuts the remaining elements roughly in half. The number of times you can halve n before reaching 1 is exactly log base 2 of n -- that count of halving steps is what O(log n) describes.",
                    "example": "16 -> 8 -> 4 -> 2 -> 1 is 4 halving steps, and log2(16) = 4. Doubling the array to 32 only adds one more step (5).",
                    "analogy": "Think of folding a piece of paper in half repeatedly -- you reach a single layer's width in very few folds, because each fold removes half of what's left, not a fixed amount.",
                    "mini_question": "If the search space becomes 16 -> 8 -> 4 -> 2 -> 1, how many reduction steps were needed?",
                    "mini_answer": "4 steps -- and the step count only grows by 1 every time the input size doubles, which is exactly logarithmic growth.",
                },
            },
        ],
        "misconceptions": [
            {
                "claim": "Binary search works on any array, sorted or not.",
                "trigger": ["works on any array", "any array", "doesn't need to be sorted", "unsorted array works"],
                "explanation": "Without a sorted order, comparing against the middle element gives no information about which half could contain the target, so the algorithm's core trick breaks down.",
            },
            {
                "claim": "Binary search is O(log n) simply because it checks the middle element.",
                "trigger": ["fast because it checks the middle", "o(log n) because middle", "log n because it picks the middle"],
                "explanation": "Checking the middle is *how* the algorithm decides what to discard, but the complexity comes from counting how many halving steps are needed to shrink n down to 1.",
            },
        ],
    },
    "arrays": {
        "name": "Arrays",
        "subject": "Data Structures",
        "difficulty": "beginner",
        "description": "A fixed-layout collection of elements stored in contiguous memory, accessed by index.",
        "learning_objective": "Explain how arrays store data, why index access is O(1), and why insertion in the middle is costly.",
        "prerequisites": [],
        "key_points": [
            {
                "label": "contiguous_memory",
                "claim": "Array elements are stored in contiguous (back-to-back) memory locations.",
                "keywords": ["contiguous", "consecutive memory", "back to back", "next to each other in memory", "same block of memory"],
                "type": "fact", "importance": 3,
                "challenge": "Why does an array need contiguous memory instead of scattered memory like a linked list?",
                "repair": {
                    "explanation": "Because elements sit one after another in memory, the address of any element can be calculated directly from the start address plus an offset -- there's nothing to walk through to find it.",
                    "example": "If an array starts at address 1000 and each element takes 4 bytes, index 5 sits at exactly 1000 + 5*4 = 1020.",
                    "analogy": "It's like numbered seats in a single row of a theater -- you can walk directly to seat 12 without checking every seat before it.",
                    "mini_question": "If an array starts at address 200 and each element is 8 bytes, where is index 3?",
                    "mini_answer": "200 + 3*8 = 224.",
                },
            },
            {
                "label": "index_access",
                "claim": "Accessing an element by index is O(1) because the address can be computed directly.",
                "keywords": ["o(1)", "constant time", "direct access", "random access", "index access is fast"],
                "type": "reasoning", "importance": 3,
                "challenge": "Why is arr[500] just as fast to access as arr[0], regardless of array size?",
                "repair": {
                    "explanation": "The address formula (base address + index * element size) takes the same single calculation no matter which index you ask for, so access time doesn't grow with array size.",
                    "example": "Looking up arr[0] and arr[999999] both take one multiplication and one addition.",
                    "analogy": "It's like knowing a house's street number lets you walk straight there, whether it's house #2 or house #200.",
                    "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "insertion_cost",
                "claim": "Inserting into the middle of an array is costly because later elements must shift.",
                "keywords": ["shift", "shifting elements", "move elements", "insert in the middle", "expensive insertion"],
                "type": "reasoning", "importance": 2,
                "challenge": "Why does inserting one element near the start of a large array take longer than inserting one at the end?",
                "repair": {
                    "explanation": "Because slots are contiguous with no gaps, making room for a new element means physically shifting every element after the insertion point one position over.",
                    "example": "Inserting at index 1 in [1,2,3,4,5] requires moving 2,3,4,5 each one slot to the right.",
                    "analogy": "It's like squeezing into a full row of theater seats -- everyone from your seat onward has to scoot over by one.",
                    "mini_question": "Which is cheaper: inserting at the end of an array, or at the beginning? Why?",
                    "mini_answer": "Inserting at the end is cheaper (often O(1)) because no existing elements need to shift.",
                },
            },
        ],
        "misconceptions": [
            {
                "claim": "Arrays and linked lists have the same performance for all operations.",
                "trigger": ["same as linked list", "no difference from linked list", "arrays and linked lists are basically the same"],
                "explanation": "Contiguous memory makes array indexing instant but insertion costly; linked lists trade off in the opposite direction.",
            },
        ],
    },
    "stack": {
        "name": "Stack",
        "subject": "Data Structures",
        "difficulty": "beginner",
        "description": "A Last-In-First-Out (LIFO) collection that only allows adding and removing from one end.",
        "learning_objective": "Explain LIFO ordering, push/pop, and a real use case such as function call tracking.",
        "prerequisites": [],
        "key_points": [
            {
                "label": "lifo_order",
                "claim": "A stack follows Last-In-First-Out order: the most recently added element is removed first.",
                "keywords": ["lifo", "last in first out", "last element added is removed first", "most recent added removed first"],
                "type": "fact", "importance": 3,
                "challenge": "If you push 1, 2, then 3 onto a stack, in what order do they come off, and why?",
                "repair": {
                    "explanation": "A stack only exposes one end (the 'top'). Since every push and pop happens at that same end, the last thing pushed is sitting right on top and is necessarily the first thing popped.",
                    "example": "Push 1, 2, 3 -> popping gives 3, then 2, then 1.",
                    "analogy": "It's like a stack of plates -- you place new plates on top, and you always take the top plate off first.",
                    "mini_question": "You push A, then B, then C. What does the first pop return?",
                    "mini_answer": "C -- it was the last one pushed.",
                },
            },
            {
                "label": "push_pop_operations",
                "claim": "The two core operations are push (add to top) and pop (remove from top).",
                "keywords": ["push", "pop", "add to top", "remove from top", "top of the stack"],
                "type": "fact", "importance": 2,
                "challenge": "What operation would you use to look at the top element without removing it?",
                "repair": {
                    "explanation": "Push and pop only ever touch the top of the stack -- there is deliberately no operation to insert or remove from the middle.",
                    "example": "push(5) places 5 on top; pop() removes and returns whatever is currently on top.",
                    "analogy": None, "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "call_stack_use_case",
                "claim": "Stacks are used to track function calls (the call stack), including recursive calls.",
                "keywords": ["call stack", "function calls", "recursion uses a stack", "tracks function calls"],
                "type": "reasoning", "importance": 2,
                "challenge": "Why is a stack (rather than a queue) the right structure for tracking function calls?",
                "repair": {
                    "explanation": "A function must finish and return before the function that called it can continue -- meaning the most recently called function is always the next one to finish. That's exactly LIFO order.",
                    "example": "If A() calls B() which calls C(), C() finishes first, then B(), then A().",
                    "analogy": "It mirrors a stack of tasks where you fully finish the task you most recently started before returning to the one before it.",
                    "mini_question": None, "mini_answer": None,
                },
            },
        ],
        "misconceptions": [
            {
                "claim": "A stack lets you remove any element you want, not just the top.",
                "trigger": ["remove any element", "access any element directly", "remove from the middle"],
                "explanation": "Allowing arbitrary removal would break the LIFO guarantee that defines a stack.",
            },
        ],
    },
    "recursion": {
        "name": "Recursion",
        "subject": "Algorithms",
        "difficulty": "intermediate",
        "description": "A function that solves a problem by calling itself on a smaller version of the same problem.",
        "learning_objective": "Explain base cases, the recursive case, and why every recursive call must move toward the base case.",
        "prerequisites": ["stack"],
        "key_points": [
            {
                "label": "base_case",
                "claim": "Every recursive function needs a base case that stops the recursion.",
                "keywords": ["base case", "stopping condition", "terminates", "stops recursion", "exit condition"],
                "type": "fact", "importance": 3,
                "challenge": "What happens if a recursive function has no base case, or it's never reached?",
                "repair": {
                    "explanation": "The base case is the only thing that stops the chain of calls. Without one, each call keeps spawning another call indefinitely, exhausting the call stack.",
                    "example": "factorial(n) needs 'if n == 0: return 1' -- without it, factorial(5) calls factorial(4), factorial(3), ... forever.",
                    "analogy": "It's like nesting dolls with no smallest, solid doll to stop at -- you'd keep opening dolls forever.",
                    "mini_question": "What error typically occurs if a recursive function never reaches its base case?",
                    "mini_answer": "A stack overflow -- the call stack fills up with unfinished calls until it runs out of space.",
                },
            },
            {
                "label": "recursive_case",
                "claim": "The recursive case breaks the problem into a smaller version of itself and calls the function again.",
                "keywords": ["recursive case", "calls itself", "smaller subproblem", "calls itself with a smaller"],
                "type": "fact", "importance": 2,
                "challenge": "In factorial(n) = n * factorial(n-1), what exactly is 'smaller' about the subproblem?",
                "repair": {
                    "explanation": "The recursive case must call the function on an input that is strictly closer to the base case, or the recursion never makes progress toward stopping.",
                    "example": "factorial(5) calls factorial(4), which calls factorial(3), each one step closer to the base case of 0.",
                    "analogy": None, "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "call_stack_growth",
                "claim": "Each recursive call adds a new frame to the call stack until the base case is hit, then frames unwind.",
                "keywords": ["call stack grows", "stack frame", "unwind", "unwinds", "each call adds a frame"],
                "type": "reasoning", "importance": 2,
                "challenge": "Why does deep recursion risk a stack overflow, and what does 'unwinding' mean afterward?",
                "repair": {
                    "explanation": "Every call that hasn't returned yet needs its own frame kept on the stack. Calls pile up frame by frame until the base case returns, then each frame resolves and pops off in reverse order.",
                    "example": "factorial(3) keeps factorial(3,2,1,0) all on the stack at once before factorial(0) returns and the stack unwinds.",
                    "analogy": "It's like putting each unfinished task on top of a to-do pile -- you can't cross off task 1 until everything piled on top is finished.",
                    "mini_question": None, "mini_answer": None,
                },
            },
        ],
        "misconceptions": [
            {
                "claim": "Recursion is always more efficient than a loop.",
                "trigger": ["recursion is always faster", "always more efficient than a loop", "recursion is always better than iteration"],
                "explanation": "Recursive calls pay the ongoing memory cost of the call stack, so an iterative version often does the same work with less overhead.",
            },
        ],
    },
    "time-complexity": {
        "name": "Time Complexity",
        "subject": "Algorithms",
        "difficulty": "beginner",
        "description": "A way of describing how an algorithm's running time grows as the input size grows, using Big-O notation.",
        "learning_objective": "Explain what Big-O measures, and why constant factors and small inputs don't matter for it.",
        "prerequisites": [],
        "key_points": [
            {
                "label": "growth_not_speed",
                "claim": "Big-O describes how runtime grows with input size, not the exact runtime in seconds.",
                "keywords": ["growth rate", "how it grows", "not actual time", "describes growth", "scales with input"],
                "type": "fact", "importance": 3,
                "challenge": "Why can two algorithms both be O(n) even if one is measurably slower in practice?",
                "repair": {
                    "explanation": "Big-O ignores constant factors and hardware differences on purpose -- it only describes the shape of the growth curve as n gets large.",
                    "example": "One O(n) algorithm might do 2 operations per element and another 10, but both still double their work when n doubles.",
                    "analogy": None, "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "worst_case_focus",
                "claim": "Big-O most commonly describes the worst-case number of operations.",
                "keywords": ["worst case", "upper bound", "worst-case scenario"],
                "type": "fact", "importance": 2,
                "challenge": "Why do we usually care about the worst case rather than the best or average case?",
                "repair": {
                    "explanation": "The worst case gives a guarantee -- a promise that the algorithm will never take longer than this, no matter the input.",
                    "example": "Linear search is O(n) worst-case because the target might be the very last element checked.",
                    "analogy": None, "mini_question": None, "mini_answer": None,
                },
            },
            {
                "label": "dominant_term",
                "claim": "Only the fastest-growing term matters as n gets large; lower-order terms and constants are dropped.",
                "keywords": ["dominant term", "drop constants", "fastest growing term", "ignore lower order terms", "as n gets large"],
                "type": "reasoning", "importance": 3,
                "challenge": "Why is an algorithm that does n^2 + 100n operations still called O(n^2)?",
                "repair": {
                    "explanation": "As n grows large enough, n^2 outpaces 100n by an ever-widening margin -- the term that grows fastest eventually dominates the total.",
                    "example": "At n=50, 100n looks bigger -- but by n=200, n^2=40000 dwarfs 100n=20000, and the gap only widens.",
                    "analogy": "It's like comparing a car accelerating exponentially against one at a fixed speed -- it might start behind, but given enough distance it always pulls ahead for good.",
                    "mini_question": "Between n^2 and 1000n, which term dominates as n approaches infinity?",
                    "mini_answer": "n^2 -- it eventually exceeds 1000n and keeps growing faster after that.",
                },
            },
        ],
        "misconceptions": [
            {
                "claim": "An O(n) algorithm is always faster in practice than an O(log n) algorithm for any input size.",
                "trigger": ["o(n) is always faster", "always faster than o(log n)", "linear is always faster in practice"],
                "explanation": "Big-O is an asymptotic comparison -- it only guarantees which algorithm wins as n becomes large, not for every specific input size.",
            },
        ],
    },
}

# ===========================================================================
# 2. TEXT UTILITIES (preprocessing, section 6.5 steps 1-3)
# ===========================================================================
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "for", "and", "or", "but", "so", "it", "its", "this", "that", "these", "those",
    "with", "as", "at", "by", "from", "we", "you", "i", "they", "he", "she", "then", "than",
}

REASONING_CONNECTORS = [
    "because", "since", "therefore", "so that", "as a result", "due to", "this means",
    "which means", "the reason", "why", "hence", "thus", "that's why", "causes", "leads to", "results in",
]

UNCERTAINTY_MARKERS = ["i think", "maybe", "probably", "not sure", "i guess", "i believe", "possibly", "might be"]


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-\./]", " ", text)
    return re.sub(r"\s+", " ", text)


def _phrase_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def keyword_hits(clean: str, keywords: list) -> list:
    """Returns the subset of keywords present verbatim or as a close fuzzy match."""
    hits = []
    tokens = clean.split(" ")
    for kw in keywords:
        kw_clean = clean_text(kw)
        if kw_clean in clean:
            hits.append(kw)
            continue
        window_size = max(1, len(kw_clean.split(" ")))
        for i in range(len(tokens) - window_size + 1):
            window = " ".join(tokens[i:i + window_size])
            if _phrase_similarity(window, kw_clean) >= 0.82:
                hits.append(kw)
                break
    return hits


def has_reasoning_cue(clean: str) -> bool:
    return any(c in clean for c in REASONING_CONNECTORS)


def has_uncertainty_cue(clean: str) -> bool:
    return any(m in clean for m in UNCERTAINTY_MARKERS)


# ===========================================================================
# 3. EVALUATION (section 6.5 steps 4-6)
# ===========================================================================
def evaluate_explanation(slug: str, explanation_text: str) -> dict:
    """
    Returns:
      {
        "understanding": int, "accuracy": int, "reasoning": int,
        "completeness": int, "application": int, "status": str,
        "matched_points": [str], "gaps": [ {gap_type, target_label,
        is_misconception, misconception_index, severity, description} ]
      }
    """
    concept = CONCEPTS[slug]
    clean = clean_text(explanation_text)

    scored = []
    for kp in concept["key_points"]:
        hits = keyword_hits(clean, kp["keywords"])
        matched = bool(hits)
        reasoning_ok = True
        if kp["type"] == "reasoning" and matched:
            reasoning_ok = has_reasoning_cue(clean)
        scored.append({"kp": kp, "matched": matched, "reasoning_ok": reasoning_ok})

    total_weight = sum(s["kp"]["importance"] for s in scored) or 1
    matched_weight = sum(s["kp"]["importance"] for s in scored if s["matched"])
    completeness = round(100 * matched_weight / total_weight)

    reasoning_pts = [s for s in scored if s["kp"]["type"] == "reasoning"]
    if reasoning_pts:
        r_weight = sum(s["kp"]["importance"] for s in reasoning_pts)
        r_ok = sum(s["kp"]["importance"] for s in reasoning_pts if s["matched"] and s["reasoning_ok"])
        reasoning = round(100 * r_ok / r_weight)
    else:
        reasoning = completeness

    triggered_misconceptions = []
    for idx, m in enumerate(concept["misconceptions"]):
        if keyword_hits(clean, m["trigger"]):
            triggered_misconceptions.append((idx, m))

    accuracy = max(0, completeness - 25 * len(triggered_misconceptions))
    understanding = round(0.4 * completeness + 0.35 * reasoning + 0.25 * accuracy)

    critical = [s for s in scored if s["kp"]["importance"] >= 3]
    application = round(100 * sum(1 for s in critical if s["matched"]) / len(critical)) if critical else completeness

    gaps = []
    for s in scored:
        kp = s["kp"]
        if not s["matched"]:
            gaps.append({
                "gap_type": "KNOWLEDGE_GAP", "target_label": kp["label"], "is_misconception": False,
                "misconception_index": None, "severity": kp["importance"],
                "description": f"The explanation never mentions: {kp['claim']}",
            })
        elif kp["type"] == "reasoning" and not s["reasoning_ok"]:
            gaps.append({
                "gap_type": "REASONING_GAP", "target_label": kp["label"], "is_misconception": False,
                "misconception_index": None, "severity": kp["importance"],
                "description": f"You mention '{kp['label'].replace('_', ' ')}' but don't explain *why* it's true: {kp['claim']}",
            })

    for idx, m in triggered_misconceptions:
        gaps.append({
            "gap_type": "MISCONCEPTION", "target_label": None, "is_misconception": True,
            "misconception_index": idx, "severity": 3,
            "description": f"Incorrect claim detected: {m['claim']} -> {m['explanation']}",
        })

    if has_uncertainty_cue(clean):
        gaps.append({
            "gap_type": "INCOMPLETE_EXPLANATION", "target_label": None, "is_misconception": False,
            "misconception_index": None, "severity": 1,
            "description": "The explanation hedges on part of the answer (e.g. 'I think', 'not sure').",
        })

    priority = {"MISCONCEPTION": 3, "REASONING_GAP": 2, "KNOWLEDGE_GAP": 1, "INCOMPLETE_EXPLANATION": 0}
    gaps.sort(key=lambda g: (priority.get(g["gap_type"], 0), g["severity"]), reverse=True)

    if understanding >= 85 and not any(g["gap_type"] == "MISCONCEPTION" for g in gaps):
        status = "understood"
    elif understanding >= 45:
        status = "partially_understood"
    else:
        status = "not_understood"

    return {
        "understanding": understanding, "accuracy": accuracy, "reasoning": reasoning,
        "completeness": completeness, "application": application, "status": status,
        "matched_points": [s["kp"]["label"] for s in scored if s["matched"]],
        "gaps": gaps,
    }


# ===========================================================================
# 4. ADAPTIVE QUESTION GENERATION (section 6.5 step 7)
# ===========================================================================
def _difficulty_for(severity: int) -> str:
    return {1: "easy", 2: "medium", 3: "hard"}.get(severity, "medium")


# Appended to the base challenge question when the student rated their own
# confidence 4-5 (see level_for_confidence()). Two variants so fact-style and
# reasoning-style gaps each get a push that actually fits the kind of gap.
_ADVANCED_SUFFIX = {
    "reasoning": " Go a step further: what would break if this reasoning didn't hold, and how does it connect to a related concept you know?",
    "fact": " Go a step further: describe a concrete scenario where forgetting this would cause a subtle bug, not just a wrong answer.",
}


def level_for_confidence(self_confidence) -> str:
    """1-3 self-rated confidence -> basic questions; 4-5 -> moderate/high (advanced)."""
    return "advanced" if (self_confidence or 0) >= 4 else "basic"


def generate_challenge(slug: str, gap: dict, level: str = "basic") -> dict:
    concept = CONCEPTS[slug]

    if gap["is_misconception"]:
        question = f"You said something close to: \"{concept['misconceptions'][gap['misconception_index']]['claim']}\". Is that always true? Explain what actually happens and why."
        if level == "advanced":
            question += " Also explain the specific situation where someone would be misled into believing this."
        return {
            "question": question, "target_label": None, "is_misconception": True,
            "misconception_index": gap["misconception_index"],
            "difficulty": _difficulty_for(gap["severity"]), "level": level,
        }

    if gap["target_label"] is None:
        # e.g. INCOMPLETE_EXPLANATION -- hedging detected but no single key point to target
        question = "Which part of your explanation are you least sure about? Try stating it definitively, without hedging."
        if level == "advanced":
            question += " Then justify it as if someone were about to challenge you on it."
        return {
            "question": question, "target_label": None, "is_misconception": False,
            "misconception_index": None, "difficulty": _difficulty_for(gap["severity"]), "level": level,
        }

    kp = next(k for k in concept["key_points"] if k["label"] == gap["target_label"])
    question = kp["challenge"]
    if level == "advanced":
        question += _ADVANCED_SUFFIX.get(kp["type"], _ADVANCED_SUFFIX["fact"])
    return {
        "question": question, "target_label": kp["label"], "is_misconception": False,
        "misconception_index": None, "difficulty": _difficulty_for(gap["severity"]), "level": level,
    }


# ===========================================================================
# 5. GRADING + REPAIR GENERATION (section 6.5 step 8)
# ===========================================================================
def grade_challenge_answer(slug: str, target_label, is_misconception: bool, answer_text: str) -> str:
    """Returns CORRECT / PARTIAL / INCORRECT."""
    concept = CONCEPTS[slug]
    clean = clean_text(answer_text)

    if is_misconception or target_label is None:
        has_cue = has_reasoning_cue(clean)
        has_content = len(clean.split(" ")) >= 6
        if has_cue and has_content:
            return "CORRECT"
        return "PARTIAL" if has_content else "INCORRECT"

    kp = next((k for k in concept["key_points"] if k["label"] == target_label), None)
    if kp is None:
        return "INCORRECT"

    if not keyword_hits(clean, kp["keywords"]):
        return "INCORRECT"
    if kp["type"] == "reasoning" and not has_reasoning_cue(clean):
        return "PARTIAL"
    return "CORRECT"


def generate_repair(slug: str, target_label, is_misconception: bool, misconception_index=None) -> dict:
    concept = CONCEPTS[slug]

    if is_misconception:
        m = concept["misconceptions"][misconception_index]
        return {"repair_type": "EXPLANATION", "content": {
            "explanation": m["explanation"], "example": None, "analogy": None,
            "mini_question": None, "mini_answer": None,
        }}

    kp = next((k for k in concept["key_points"] if k["label"] == target_label), None)
    if kp is None:
        return {"repair_type": "EXPLANATION", "content": {
            "explanation": "Re-read the core definition and try naming each step out loud before typing it.",
            "example": None, "analogy": None, "mini_question": None, "mini_answer": None,
        }}
    return {"repair_type": "EXPLANATION", "content": kp["repair"]}


def compose_combined_answer(slug: str, target_label, is_misconception: bool, misconception_index, grade: str, user_answer_text: str) -> str:
    """
    Builds the "TeachBack's answer" shown on the final Verification screen:
    a short synthesis that acknowledges what the student said and folds in
    the authoritative explanation, rather than just dumping the repair text
    on its own.
    """
    concept = CONCEPTS[slug]
    trimmed = user_answer_text.strip()
    if len(trimmed) > 220:
        trimmed = trimmed[:217].rstrip() + "..."

    if is_misconception:
        m = concept["misconceptions"][misconception_index]
        explanation = m["explanation"]
    else:
        kp = next((k for k in concept["key_points"] if k["label"] == target_label), None)
        explanation = kp["repair"]["explanation"] if kp else "Here's the key idea to hold onto."

    if grade == "CORRECT":
        opener = f"You said: \"{trimmed}\" -- that's right."
        return f"{opener} To put it together with the full picture: {explanation}"
    elif grade == "PARTIAL":
        opener = f"You said: \"{trimmed}\" -- you're partly there."
        return f"{opener} Here's the piece to add: {explanation}"
    else:
        opener = f"You said: \"{trimmed}\"." if trimmed else "You left this blank."
        return f"{opener} Here's the full answer: {explanation}"


# ===========================================================================
# 6. VERIFICATION (section 6.9)
# ===========================================================================
VERIFY_THRESHOLDS = {"understanding": 75, "accuracy": 80, "reasoning": 70}


def check_verification(evaluation: dict, resolved_target_gap: bool = True) -> str:
    has_misconception = any(g["gap_type"] == "MISCONCEPTION" for g in evaluation["gaps"])
    cleared = (
        evaluation["understanding"] >= VERIFY_THRESHOLDS["understanding"]
        and evaluation["accuracy"] >= VERIFY_THRESHOLDS["accuracy"]
        and evaluation["reasoning"] >= VERIFY_THRESHOLDS["reasoning"]
        and not has_misconception
        and resolved_target_gap
    )
    return "VERIFIED" if cleared else "NEEDS_REINFORCEMENT"


# ===========================================================================
# PUBLIC API for backend.py
# ===========================================================================
def list_concepts_meta() -> list:
    """Lightweight metadata for the concept library (no key points/misconceptions)."""
    return [
        {
            "slug": slug, "name": c["name"], "subject": c["subject"],
            "difficulty": c["difficulty"], "description": c["description"],
        }
        for slug, c in CONCEPTS.items()
    ]


def list_concepts_meta_by_slug() -> dict:
    """Same metadata as list_concepts_meta(), keyed by slug, slug omitted from the value
    (handy for `Concept(slug=slug, **meta)` style construction in backend.py)."""
    return {
        slug: {"name": c["name"], "subject": c["subject"], "difficulty": c["difficulty"], "description": c["description"]}
        for slug, c in CONCEPTS.items()
    }


def get_concept_detail(slug: str) -> dict:
    c = CONCEPTS[slug]
    return {
        "slug": slug, "name": c["name"], "subject": c["subject"], "difficulty": c["difficulty"],
        "description": c["description"], "learning_objective": c["learning_objective"],
        "prerequisites": c["prerequisites"],
        "key_points": [{"label": k["label"], "claim": k["claim"], "type": k["type"], "importance": k["importance"], "challenge": k["challenge"]} for k in c["key_points"]],
    }


# ===========================================================================
# 7. QUIZ MODE (multiple-choice questions, used by the quiz flow in backend.py)
# ===========================================================================
# Flow: confidence 1-3 -> "basic" questions first, 4-5 -> "advanced" first.
# When the preferred level runs out, the other level follows, so a learner
# can keep going until every question for the concept has been answered.
#
# Each row: (level, question, [4 options], index_of_correct_option, explanation)
QUIZ_PASS_PERCENT = 70

_RAW_QUESTIONS = {
    "binary-search": [
        ("basic", "What must be true about an array before binary search can be used on it?",
         ["It must be sorted", "It must contain no duplicates", "It must have an even number of elements", "It must contain only integers"], 0,
         "Binary search discards half the data by comparing with the middle value. That only works if the data is in sorted order."),
        ("basic", "Which element does binary search compare with the target at each step?",
         ["The first element", "The middle element", "The last element", "A random element"], 1,
         "Checking the middle lets the algorithm throw away half of the remaining elements in a single comparison."),
        ("basic", "The target is larger than the middle element. What happens next?",
         ["The upper half is discarded", "Only the middle element is discarded", "The lower half (including the middle) is discarded", "The search restarts from the beginning"], 2,
         "Because the array is sorted, everything at or below the middle is smaller than the target, so it can't contain it."),
        ("basic", "What is the time complexity of binary search?",
         ["O(n)", "O(n^2)", "O(1)", "O(log n)"], 3,
         "The search space is halved every step, and you can halve n only about log2(n) times before reaching 1."),
        ("advanced", "Searching for 23 in [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] (indices 0-9, mid = (low + high) // 2). How many middle-element comparisons are needed?",
         ["2", "3", "4", "6"], 1,
         "Mid=index 4 (16) -> go right; mid=index 7 (56) -> go left; mid=index 5 (23) -> found. That is 3 comparisons."),
        ("advanced", "Roughly how many steps does binary search need, at worst, for 1,000,000 sorted elements?",
         ["About 1,000", "About 20", "About 500,000", "About 10"], 1,
         "log2(1,000,000) is about 20, so at most around 20 halvings are needed."),
        ("advanced", "Someone says: \"Binary search is O(log n) because it checks the middle element.\" What is wrong with that reasoning?",
         ["Nothing, it is correct", "Checking the middle itself takes log n time", "Checking the middle only decides what to discard; the log n comes from counting how many halvings it takes to shrink n to 1", "Binary search is actually O(n)"], 2,
         "The middle check is the mechanism. The complexity comes from repeated halving: the number of halvings from n to 1 is log2(n)."),
        ("advanced", "The input size doubles from n to 2n. How many extra steps does binary search need in the worst case?",
         ["One more step", "Twice as many steps", "n more steps", "No extra steps"], 0,
         "Doubling the input adds just one more halving, which is exactly what logarithmic growth means."),
    ],
    "arrays": [
        ("basic", "How are the elements of an array stored in memory?",
         ["In contiguous (back-to-back) memory locations", "Linked together by pointers", "Scattered randomly", "In hash buckets"], 0,
         "Arrays occupy one continuous block of memory, which is what makes index access so fast."),
        ("basic", "What is the time complexity of accessing arr[i] by index?",
         ["O(n)", "O(1)", "O(log n)", "O(n^2)"], 1,
         "The address is computed directly (start + index * element size), so access time doesn't depend on array size."),
        ("basic", "Why is inserting an element at the start of an array costly?",
         ["The array must be re-sorted", "Every later element must shift one position over", "Indexes stop working", "A new array must always be created on another machine"], 1,
         "There are no gaps in contiguous memory, so making room means shifting all following elements."),
        ("basic", "Which is usually cheaper in an array?",
         ["Inserting at the beginning", "Inserting in the middle", "Inserting at the end", "They all cost exactly the same"], 2,
         "Inserting at the end needs no shifting of existing elements (often O(1))."),
        ("advanced", "An array starts at memory address 500 and each element takes 4 bytes. At what address is index 10?",
         ["510", "540", "504", "544"], 1,
         "address = 500 + 10 * 4 = 540."),
        ("advanced", "Why is arr[999999] just as fast to reach as arr[0]?",
         ["Arrays are always sorted", "The computer caches every element", "One formula (base + index * size) gives the address, no matter which index", "Arrays search internally with binary search"], 2,
         "The same single calculation works for any index, so access time doesn't grow with the array size."),
        ("advanced", "An array has n elements. How many elements must be shifted to insert a new element at index 0?",
         ["None", "1", "log n", "All n existing elements"], 3,
         "Every existing element has to move one slot to the right to free up index 0."),
        ("advanced", "Someone claims arrays and linked lists perform identically for all operations. Which is a counterexample?",
         ["Indexed access is O(1) in an array but O(n) in a linked list", "Both need contiguous memory", "Arrays can't store numbers", "Linked lists allow O(1) access by index"], 0,
         "Contiguous memory gives arrays instant indexing; linked lists must walk node by node."),
    ],
    "stack": [
        ("basic", "What ordering does a stack follow?",
         ["FIFO (first in, first out)", "LIFO (last in, first out)", "Sorted order", "Random order"], 1,
         "The most recently added element sits on top and is the first one removed."),
        ("basic", "What does the push operation do?",
         ["Removes the top element", "Looks at the top element", "Adds an element to the top", "Removes the bottom element"], 2,
         "push places a new element on top of the stack."),
        ("basic", "You push 1, then 2, then 3. What does the first pop return?",
         ["1", "2", "It causes an error", "3"], 3,
         "3 was pushed last, so it is on top and comes off first."),
        ("basic", "Which is a classic use of a stack?",
         ["Tracking function calls (the call stack)", "Serving customers in arrival order", "Searching sorted data", "Round-robin load balancing"], 0,
         "The most recently called function must finish first, which is LIFO behaviour."),
        ("advanced", "Operations: push A, push B, push C, pop, push D, pop, pop. In what order are values popped?",
         ["C, B, A", "C, D, B", "C, D, A", "A, B, D"], 1,
         "Stack after pushes: A,B,C. pop -> C. push D -> A,B,D. pop -> D. pop -> B."),
        ("advanced", "Why is a stack, not a queue, the right structure for tracking function calls?",
         ["Queues are slower", "Stacks always use less memory", "There is no real reason", "The most recently called function must finish first, which is LIFO"], 3,
         "If A calls B which calls C, then C finishes first, then B, then A."),
        ("advanced", "Which operation lets you look at the top element without removing it?",
         ["pop", "peek", "push", "clear"], 1,
         "peek (or top) reads the top element and leaves the stack unchanged."),
        ("advanced", "Why is a stack a natural fit for checking balanced brackets like \"( [ ] )\"?",
         ["Stacks sort characters", "Brackets are numbers", "The most recently opened bracket must be the first one closed", "Queues can't hold characters"], 2,
         "Each closing bracket must match the latest unmatched opening bracket, which is exactly the top of a stack."),
    ],
    "recursion": [
        ("basic", "What is a base case?",
         ["The condition that stops the recursion", "The first call to the function", "A loop counter", "The function's return type"], 0,
         "The base case is the simplest input that is answered directly, without another recursive call."),
        ("basic", "In the recursive case, a function calls itself on...",
         ["The exact same input", "A smaller version of the problem", "A larger input", "A completely different function"], 1,
         "Each call must move closer to the base case, otherwise the recursion never ends."),
        ("basic", "What error typically occurs when a recursive function never reaches its base case?",
         ["Syntax error", "Division by zero", "Stack overflow", "It quietly returns 0"], 2,
         "Calls keep piling up on the call stack until it runs out of space."),
        ("basic", "For factorial(n) = n * factorial(n-1), which is the usual base case?",
         ["factorial(10) = 10", "factorial(1) = n", "factorial(-1) = 1", "factorial(0) = 1"], 3,
         "factorial(0) = 1 stops the chain of calls and gives the multiplications something to start from."),
        ("advanced", "For f(n) = 1 if n == 0 else n * f(n-1), how many calls are made in total (including the first) for f(4)?",
         ["4", "5", "6", "24"], 1,
         "f(4), f(3), f(2), f(1), f(0) = 5 calls."),
        ("advanced", "Why does very deep recursion risk a stack overflow?",
         ["Each call copies the whole program", "Base cases use a lot of memory", "Each unfinished call keeps its own frame on the call stack", "The interpreter forbids recursion beyond 10 calls"], 2,
         "Every call that hasn't returned yet needs a frame, so frames pile up until the base case is hit."),
        ("advanced", "Someone says recursion is always more efficient than a loop. What is the best counterpoint?",
         ["Recursion is always slower by a fixed factor", "Recursive calls cost call-stack memory, so an iterative version often does the same work with less overhead", "Loops can't solve the same problems", "Recursion doesn't need a base case"], 1,
         "Recursion can be elegant, but it isn't automatically efficient; call-stack overhead is a real cost."),
        ("advanced", "What does \"unwinding\" mean in recursion?",
         ["The recursion starts over", "The base case is skipped", "The stack is cleared before starting", "After the base case returns, each pending call finishes and pops off in reverse order"], 3,
         "Once the base case returns, the waiting calls resolve one by one, the most recent first."),
    ],
    "time-complexity": [
        ("basic", "What does Big-O notation describe?",
         ["The exact runtime in seconds", "How runtime grows as the input size grows", "The amount of disk space used", "The number of lines in the code"], 1,
         "Big-O is about the shape of the growth curve, not exact timings."),
        ("basic", "Which case does Big-O most commonly describe?",
         ["Best case", "Average case only", "Worst case", "A random case"], 2,
         "The worst case gives a guarantee: the algorithm will never do more work than this."),
        ("basic", "An algorithm does n^2 + 100n operations. Its Big-O is...",
         ["O(n^2)", "O(n)", "O(100n)", "O(n^3)"], 0,
         "As n grows, n^2 dominates 100n, so only the fastest-growing term is kept."),
        ("basic", "Two algorithms are both O(n), but one is measurably slower. Why is that possible?",
         ["One of them must really be O(n^2)", "Big-O ignores constant factors", "That is impossible", "Big-O measures seconds"], 1,
         "One may do 10 operations per element and the other 2. Both still double their work when n doubles."),
        ("advanced", "Which ordering goes from slowest-growing to fastest-growing?",
         ["O(n) < O(1) < O(log n) < O(n^2)", "O(1) < O(n) < O(log n) < O(n^2)", "O(1) < O(log n) < O(n) < O(n^2)", "O(log n) < O(1) < O(n^2) < O(n)"], 2,
         "Constant, then logarithmic, then linear, then quadratic."),
        ("advanced", "Algorithm A does 1000n operations, algorithm B does n^2. For which input sizes is A faster?",
         ["n < 1000", "n > 1000", "Always", "Never"], 1,
         "n^2 > 1000n exactly when n > 1000. B is faster for small inputs even though A has the better Big-O."),
        ("advanced", "Someone says \"an O(n) algorithm is always faster in practice than an O(log n) one\". What is the flaw?",
         ["Nothing, it is correct", "Big-O compares growth for large n; with small inputs or large constants the result can flip", "O(log n) is slower than O(n) for all n", "Big-O only applies to sorting"], 1,
         "Big-O is an asymptotic comparison and says nothing about every specific input size."),
        ("advanced", "The input size doubles. Roughly how much more work does an O(n^2) algorithm do?",
         ["2 times", "8 times", "4 times", "n times"], 2,
         "(2n)^2 = 4n^2, so the work is about four times as large."),
    ],
}


def _build_question_bank() -> dict:
    bank = {}
    for slug, rows in _RAW_QUESTIONS.items():
        counters = {"basic": 0, "advanced": 0}
        items = []
        for level, question, options, answer, explanation in rows:
            counters[level] += 1
            items.append({
                "id": f"{slug}-{level[0]}{counters[level]}", "level": level, "question": question,
                "options": options, "answer": answer, "explanation": explanation,
            })
        bank[slug] = items
    return bank


QUESTION_BANK = _build_question_bank()


def get_question_pool(slug: str, level: str) -> list:
    """Ordered questions for a session: the preferred level first, then the other level."""
    bank = QUESTION_BANK.get(slug, [])
    return [q for q in bank if q["level"] == level] + [q for q in bank if q["level"] != level]


def get_question(slug: str, question_id: str):
    return next((q for q in QUESTION_BANK.get(slug, []) if q["id"] == question_id), None)


def quiz_verdict(percent: float) -> str:
    return "VERIFIED" if percent >= QUIZ_PASS_PERCENT else "NEEDS_REINFORCEMENT"
