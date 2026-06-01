import numpy as np
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


class ClassicalSVM:
    """
    Classical Support Vector Machine using RBF (Radial Basis Function) kernel.

    This is the strong classical baseline that competes with quantum methods.
    The RBF kernel maps data into an infinite-dimensional space where classes
    can be separated by a hyperplane.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = "scale",
        kernel: str = "rbf",
        random_state: int = 42,
    ):
        """
        Initialize Classical SVM.

        Args:
            C: Regularization parameter. Controls trade-off between correct
               classification and margin maximization.
            gamma: Kernel coefficient for 'rbf'. 'scale' = 1 / (n_features * X.var())
            kernel: Kernel type. 'rbf' is recommended for non-linear separation.
            random_state: Random seed for reproducibility.

        Note: Data should be pre-scaled before passing to fit/predict for strict comparison.
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

        self.model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=random_state)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classical SVM.

        Args:
            X: Training data, shape (n_samples, n_features). Should be pre-scaled.
            y: Training labels, shape (n_samples,)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Data to predict, shape (n_samples, n_features). Should be pre-scaled.

        Returns:
            Predicted labels, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (if probability=True in SVC).
        Note: Standard SVC does not support decision_function -> probabilities conversion.
        Use decision_function scores instead.

        Args:
            X: Data to predict, shape (n_samples, n_features). Should be pre-scaled.

        Returns:
            Decision function scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.decision_function(X)


class QuantumSVM:
    """
    Quantum Support Vector Machine using Quantum Kernel (QSVC).

    Uses a Quantum Feature Map to encode classical data into a quantum Hilbert space.
    The fidelity between quantum states forms the quantum kernel matrix, which is
    passed directly to scikit-learn's SVM for classification.

    The quantum feature map enables the SVM to learn separations in quantum-transformed
    feature spaces, potentially capturing patterns classical kernels may miss.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        reps: int = 2,
        entanglement: str = "linear",
        C: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize Quantum SVM.

        Args:
            num_qubits: Number of qubits. Must match feature dimension for strict comparison.
            reps: Number of repetitions of the ZZFeatureMap. More reps = more expressiveness
                  but longer circuit and higher computational cost.
            entanglement: Type of entanglement in the feature map.
                         'linear': Chain entanglement (q0-q1-q2-...)
                         'circular': Chain with wrap-around
                         'full': All-to-all entanglement (more expressive, more expensive)
            C: Regularization parameter for the underlying SVC.
            random_state: Random seed for reproducibility.

        Note: Data should be pre-scaled before passing to fit/predict. Input features must
        match num_qubits exactly (strict 1-1 comparison with ClassicalSVM).
        """
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.random_state = random_state

        # Initialize the quantum feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_qubits, reps=reps, entanglement=entanglement
        )

        # Initialize the quantum kernel
        self.q_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # Initialize the Quantum SVM
        self.model = QSVC(quantum_kernel=self.q_kernel, C=C)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Quantum SVM.

        This computes the full quantum kernel matrix and fits the SVC.
        Note: For large datasets (>100 samples), this can be computationally expensive.

        Args:
            X: Training data, shape (n_samples, n_features). Features must equal num_qubits.
               Data should be pre-scaled.
            y: Training labels, shape (n_samples,)
        """
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"Feature dimension ({X.shape[1]}) must match num_qubits ({self.num_qubits}) "
                f"for strict 1-1 comparison with classical methods."
            )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Quantum SVM.

        Args:
            X: Data to predict, shape (n_samples, n_features). Features must equal num_qubits.
               Data should be pre-scaled.

        Returns:
            Predicted labels, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"Feature dimension ({X.shape[1]}) must match num_qubits ({self.num_qubits})."
            )
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction scores from the decision function.

        Args:
            X: Data to predict, shape (n_samples, n_features). Features must equal num_qubits.
               Data should be pre-scaled.

        Returns:
            Decision function scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"Feature dimension ({X.shape[1]}) must match num_qubits ({self.num_qubits})."
            )
        return self.model.decision_function(X)
