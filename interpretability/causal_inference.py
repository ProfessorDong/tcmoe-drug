"""
Causal Inference for Molecular Property Prediction
Implements counterfactual generation and causal analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy


class CausalAnalyzer:
    """
    Analyzes causal relationships between molecular features and properties.
    Uses intervention-based methods to identify causal effects.
    """

    def __init__(self, model, feature_names=None):
        """
        Args:
            model: Property prediction model
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names

    def compute_causal_effect(self, x, feature_idx, intervention_values):
        """
        Compute causal effect of feature intervention.

        Args:
            x: Input features [batch, features]
            feature_idx: Index of feature to intervene on
            intervention_values: List of intervention values

        Returns:
            effects: Dict with intervention results
        """
        self.model.eval()

        effects = {'baseline': [], 'interventions': {}}

        with torch.no_grad():
            # Baseline prediction
            baseline_pred = self.model(x)
            if isinstance(baseline_pred, dict):
                baseline_pred = list(baseline_pred.values())[0]
            effects['baseline'] = baseline_pred.cpu().numpy()

            # Intervention predictions
            for value in intervention_values:
                x_intervened = x.clone()
                x_intervened[:, feature_idx] = value

                pred = self.model(x_intervened)
                if isinstance(pred, dict):
                    pred = list(pred.values())[0]
                effects['interventions'][value] = pred.cpu().numpy()

        # Compute average causal effects
        effects['average_effects'] = {}
        baseline_mean = effects['baseline'].mean()
        for value, preds in effects['interventions'].items():
            effects['average_effects'][value] = preds.mean() - baseline_mean

        return effects

    def average_treatment_effect(self, x, treatment_idx, control_value=0, treatment_value=1):
        """
        Compute Average Treatment Effect (ATE).

        Args:
            x: Input features
            treatment_idx: Index of treatment feature
            control_value: Value for control group
            treatment_value: Value for treatment group

        Returns:
            ate: Average Treatment Effect
            ate_se: Standard error of ATE
        """
        self.model.eval()

        with torch.no_grad():
            # Control outcome
            x_control = x.clone()
            x_control[:, treatment_idx] = control_value
            y_control = self.model(x_control)
            if isinstance(y_control, dict):
                y_control = list(y_control.values())[0]

            # Treatment outcome
            x_treatment = x.clone()
            x_treatment[:, treatment_idx] = treatment_value
            y_treatment = self.model(x_treatment)
            if isinstance(y_treatment, dict):
                y_treatment = list(y_treatment.values())[0]

        # Compute ATE
        individual_effects = (y_treatment - y_control).cpu().numpy()
        ate = individual_effects.mean()
        ate_se = individual_effects.std() / np.sqrt(len(individual_effects))

        return ate, ate_se

    def feature_importance_via_intervention(self, x, n_samples=100):
        """
        Compute feature importance via random interventions.

        Args:
            x: Input features [1, features]
            n_samples: Number of intervention samples

        Returns:
            importance: Feature importance scores
        """
        n_features = x.size(1)
        importance = np.zeros(n_features)

        self.model.eval()

        with torch.no_grad():
            # Baseline prediction
            baseline = self.model(x)
            if isinstance(baseline, dict):
                baseline = list(baseline.values())[0]
            baseline = baseline.cpu().numpy()

            # Intervene on each feature
            for feat_idx in range(n_features):
                changes = []
                for _ in range(n_samples):
                    # Random intervention
                    x_intervened = x.clone()
                    x_intervened[:, feat_idx] = torch.randn_like(x_intervened[:, feat_idx])

                    pred = self.model(x_intervened)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]

                    changes.append(np.abs(pred.cpu().numpy() - baseline))

                importance[feat_idx] = np.mean(changes)

        # Normalize
        importance = importance / importance.sum()

        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return importance


class CounterfactualGenerator:
    """
    Generates counterfactual molecules that achieve desired property values.
    Uses optimization or search to find minimal modifications.
    """

    def __init__(self, model, generator=None):
        """
        Args:
            model: Property prediction model
            generator: Optional molecule generator for guided search
        """
        self.model = model
        self.generator = generator

    def generate_counterfactual(self, smiles, target_value, task_name=None,
                                 max_changes=3, method='gradient'):
        """
        Generate counterfactual molecule.

        Args:
            smiles: Original SMILES
            target_value: Target property value
            task_name: Task name for multi-task models
            max_changes: Maximum structural modifications
            method: 'gradient', 'random', or 'genetic'

        Returns:
            counterfactual: Modified SMILES
            changes: List of modifications made
        """
        if method == 'gradient':
            return self._gradient_counterfactual(smiles, target_value, task_name)
        elif method == 'random':
            return self._random_counterfactual(smiles, target_value, task_name, max_changes)
        elif method == 'genetic':
            return self._genetic_counterfactual(smiles, target_value, task_name, max_changes)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gradient_counterfactual(self, smiles, target_value, task_name):
        """Generate counterfactual using gradient-based optimization."""
        # This requires differentiable featurization
        # Simplified version - actual implementation would use
        # continuous relaxation of molecular structure
        raise NotImplementedError("Gradient-based counterfactuals require special featurization")

    def _random_counterfactual(self, smiles, target_value, task_name, max_changes):
        """Generate counterfactual using random modifications."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, []

        best_cf = None
        best_score = float('inf')
        changes_made = []

        # Modification operations
        operations = [
            self._add_atom,
            self._remove_atom,
            self._change_bond,
            self._add_functional_group
        ]

        for _ in range(100):  # Max attempts
            current_mol = Chem.RWMol(mol)
            current_changes = []

            for _ in range(max_changes):
                op = np.random.choice(operations)
                try:
                    new_mol, change = op(current_mol)
                    if new_mol is not None:
                        current_mol = new_mol
                        current_changes.append(change)
                except:
                    continue

            # Check if valid and evaluate
            try:
                new_smiles = Chem.MolToSmiles(current_mol.GetMol())
                pred = self._predict(new_smiles, task_name)

                score = abs(pred - target_value)
                if score < best_score:
                    best_score = score
                    best_cf = new_smiles
                    changes_made = current_changes

                if score < 0.1:  # Close enough
                    break
            except:
                continue

        return best_cf, changes_made

    def _genetic_counterfactual(self, smiles, target_value, task_name, max_changes):
        """Generate counterfactual using genetic algorithm."""
        # Population-based search
        population_size = 50
        n_generations = 20
        mutation_rate = 0.3

        # Initialize population
        population = [smiles] * population_size

        for gen in range(n_generations):
            # Evaluate fitness
            fitness = []
            for mol_smiles in population:
                try:
                    pred = self._predict(mol_smiles, task_name)
                    fitness.append(-abs(pred - target_value))
                except:
                    fitness.append(-float('inf'))

            fitness = np.array(fitness)

            # Selection
            probs = np.exp(fitness - fitness.max())
            probs = probs / probs.sum()
            selected = np.random.choice(len(population), population_size, p=probs)

            # Mutation
            new_population = []
            for idx in selected:
                mol_smiles = population[idx]
                if np.random.random() < mutation_rate:
                    mol_smiles = self._mutate_molecule(mol_smiles)
                new_population.append(mol_smiles)

            population = new_population

            # Check best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > -0.1:
                break

        best_idx = np.argmax(fitness)
        return population[best_idx], []

    def _predict(self, smiles, task_name):
        """Make prediction for SMILES."""
        # This needs to be implemented based on model interface
        raise NotImplementedError

    def _add_atom(self, mol):
        """Add atom to molecule."""
        atoms = ['C', 'N', 'O', 'F']
        atom = np.random.choice(atoms)

        # Find atom to attach to
        n_atoms = mol.GetNumAtoms()
        attach_idx = np.random.randint(n_atoms)

        new_idx = mol.AddAtom(Chem.Atom(atom))
        mol.AddBond(attach_idx, new_idx, Chem.BondType.SINGLE)

        return mol, f"Added {atom} at position {attach_idx}"

    def _remove_atom(self, mol):
        """Remove atom from molecule."""
        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 2:
            return None, None

        # Find removable atom (degree 1)
        removable = []
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetDegree() == 1:
                removable.append(i)

        if not removable:
            return None, None

        remove_idx = np.random.choice(removable)
        mol.RemoveAtom(remove_idx)

        return mol, f"Removed atom at position {remove_idx}"

    def _change_bond(self, mol):
        """Change bond type."""
        bonds = list(mol.GetBonds())
        if not bonds:
            return None, None

        bond = np.random.choice(bonds)
        bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
        new_type = np.random.choice(bond_types)

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        mol.RemoveBond(begin_idx, end_idx)
        mol.AddBond(begin_idx, end_idx, new_type)

        return mol, f"Changed bond {begin_idx}-{end_idx} to {new_type}"

    def _add_functional_group(self, mol):
        """Add functional group."""
        groups = {
            'OH': 'O',
            'NH2': 'N',
            'F': 'F',
            'Cl': 'Cl',
            'CH3': 'C'
        }

        group_name, atom = list(groups.items())[np.random.randint(len(groups))]

        n_atoms = mol.GetNumAtoms()
        attach_idx = np.random.randint(n_atoms)

        new_idx = mol.AddAtom(Chem.Atom(atom))
        mol.AddBond(attach_idx, new_idx, Chem.BondType.SINGLE)

        return mol, f"Added {group_name} at position {attach_idx}"

    def _mutate_molecule(self, smiles):
        """Apply random mutation to molecule."""
        mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
        if mol is None:
            return smiles

        operations = [self._add_atom, self._remove_atom, self._change_bond]
        op = np.random.choice(operations)

        try:
            new_mol, _ = op(mol)
            if new_mol is not None:
                return Chem.MolToSmiles(new_mol.GetMol())
        except:
            pass

        return smiles


class StructuralCausalModel:
    """
    Structural Causal Model for molecular properties.
    Models causal relationships between molecular features.
    """

    def __init__(self, feature_dim, hidden_dim=64):
        """
        Args:
            feature_dim: Number of molecular features
            hidden_dim: Hidden layer dimension
        """
        self.feature_dim = feature_dim

        # Causal graph (adjacency matrix)
        self.adjacency = nn.Parameter(torch.zeros(feature_dim, feature_dim))

        # Structural equations
        self.equations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(feature_dim)
        ])

    def forward(self, x_exogenous):
        """
        Compute endogenous variables given exogenous noise.

        Args:
            x_exogenous: Exogenous noise [batch, feature_dim]

        Returns:
            x_endogenous: Computed feature values
        """
        batch_size = x_exogenous.size(0)
        x = x_exogenous.clone()

        # Apply structural equations in topological order
        # (Simplified - assumes DAG structure)
        for i in range(self.feature_dim):
            parents = torch.sigmoid(self.adjacency[:, i])
            parent_values = x * parents
            x[:, i:i+1] = self.equations[i](parent_values) + x_exogenous[:, i:i+1]

        return x

    def intervene(self, x_exogenous, intervention_idx, intervention_value):
        """
        Perform do-calculus intervention.

        Args:
            x_exogenous: Exogenous noise
            intervention_idx: Feature to intervene on
            intervention_value: Value to set

        Returns:
            x_intervened: Values under intervention
        """
        batch_size = x_exogenous.size(0)
        x = x_exogenous.clone()

        for i in range(self.feature_dim):
            if i == intervention_idx:
                # Intervention: set value directly
                x[:, i] = intervention_value
            else:
                parents = torch.sigmoid(self.adjacency[:, i])
                parent_values = x * parents
                x[:, i:i+1] = self.equations[i](parent_values) + x_exogenous[:, i:i+1]

        return x
