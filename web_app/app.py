import os
import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory, Response
import pandas as pd
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename
import re
from collections import Counter
import json
from datetime import datetime, timezone

# Calculer les chemins de base et ajouter la racine du projet au path Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# importer les utilitaires d'exemples depuis ai_module
try:
	from ai_module import examples as ai_examples
except Exception:
	ai_examples = None

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')

# Load simple .env file from project root if present (compatible fallback, no dependency)
def _load_dotenv(path=None):
    try:
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if not os.path.isfile(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                if '=' not in ln:
                    continue
                k, v = ln.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # don't override existing environment
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

_load_dotenv()

# Prepare authentication passwords — support AUTH_PASSWORDS (comma-separated) and a separate ADMIN_PASSWORD
# Read AUTH_PASSWORDS (may be comma-separated list) — do NOT fall back to ADMIN_PASSWORD here.
_auth_passwords_raw = os.environ.get('AUTH_PASSWORDS') or ''
# Keep ADMIN raw value separately for admin-checks (may be a bcrypt hash or plain)
_ADMIN_RAW = os.environ.get('ADMIN_PASSWORD')

# Build the list of configured auth passwords (from AUTH_PASSWORDS only)
if _auth_passwords_raw:
    raw = [p.strip() for p in _auth_passwords_raw.split(',') if p.strip()]
else:
    raw = []

# Separate plain and hashed (bcrypt) entries from the AUTH_PASSWORDS list
_plain_passwords = [p for p in raw if not (p.startswith('$2a$') or p.startswith('$2b$') or p.startswith('$2y$'))]
_hashed_passwords = [p for p in raw if (p.startswith('$2a$') or p.startswith('$2b$') or p.startswith('$2y$'))]

# Whether authentication is enabled: enabled if either AUTH_PASSWORDS provided or ADMIN_PASSWORD provided
AUTH_ENABLED = bool(raw or _ADMIN_RAW)

from functools import wraps
from flask import session

# Helper to check admin password
def _is_admin_password(plaintext: str) -> (bool, str):
    """Return (True, None) if plaintext matches ADMIN_PASSWORD, (False, None) otherwise.
    If bcrypt is required but not installed, returns (False, error_message).
    """
    if not _ADMIN_RAW:
        return False, None
    # plain vs bcrypt
    if _ADMIN_RAW.startswith('$2a$') or _ADMIN_RAW.startswith('$2b$') or _ADMIN_RAW.startswith('$2y$'):
        try:
            import bcrypt
        except Exception:
            return False, 'Server misconfiguration: bcrypt not installed but ADMIN_PASSWORD is a bcrypt hash.'
        try:
            if bcrypt.checkpw(plaintext.encode('utf-8'), _ADMIN_RAW.encode('utf-8')):
                return True, None
            return False, None
        except Exception:
            return False, None
    else:
        # plain compare
        return (plaintext == _ADMIN_RAW), None

# Decorator to protect routes; defined early so it's available for decorators later
def login_required(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        if session.get('logged_in'):
            return f(*args, **kwargs)
        from flask import redirect, request
        return redirect(url_for('login', next=request.path))
    return _wrapped


def _verify_password(plaintext: str) -> (bool, str):
    """Verify provided plaintext password against configured passwords.
    Returns (True, None) if ok, or (False, error_message) if not. If bcrypt is required but not installed, returns error message explaining.
    """
    # Check plain passwords first
    for p in _plain_passwords:
        if plaintext == p:
            return True, None

    # If hashed passwords present, try bcrypt (lazy import)
    if _hashed_passwords:
        try:
            import bcrypt
        except Exception:
            return False, 'Server misconfiguration: bcrypt not installed but hashed passwords are configured. Install "bcrypt" package.'
        try:
            pb = plaintext.encode('utf-8')
            for h in _hashed_passwords:
                try:
                    if bcrypt.checkpw(pb, h.encode('utf-8')):
                        return True, None
                except Exception:
                    # ignore malformed hash
                    continue
        except Exception:
            return False, 'Erreur durant la vérification du mot de passe.'

    return False, None

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if not AUTH_ENABLED:
        # auth not enabled - inform user
        return render_template('login.html', disabled=True)
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        nxt = request.args.get('next') or request.form.get('next') or url_for('index')

        # First, verify against AUTH_PASSWORDS
        ok, err = _verify_password(pwd)
        if err:
            # _verify_password may return an explanatory error (e.g., bcrypt not installed)
            flash(err, 'danger')
            return render_template('login.html', next=nxt)

        admin_matched = False
        if not ok:
            # If not matched by AUTH_PASSWORDS, allow ADMIN_PASSWORD to authenticate
            try:
                is_admin, admin_err = _is_admin_password(pwd)
                if admin_err:
                    flash(admin_err, 'danger')
                    return render_template('login.html', next=nxt)
                if is_admin:
                    ok = True
                    admin_matched = True
            except Exception:
                # fallback: do not authenticate
                ok = False
        else:
            # If AUTH_PASSWORDS matched, additionally check whether this is the admin password
            try:
                is_admin, _ = _is_admin_password(pwd)
                if is_admin:
                    admin_matched = True
            except Exception:
                admin_matched = False

        if not ok:
            flash('Mot de passe invalide', 'danger')
            return render_template('login.html', next=nxt)

        # success: set session and is_admin flag
        session['logged_in'] = True
        if admin_matched:
            session['is_admin'] = True
        else:
            session.pop('is_admin', None)

        # assign a session id and record login event
        try:
            import uuid
            sid = str(uuid.uuid4())
            session['session_id'] = sid
            _record_login(sid, request.remote_addr, request.headers.get('User-Agent', ''))
        except Exception:
            app.logger.debug('Failed to record login event')
        flash('Connecté')
        return redirect(nxt)
    # GET
    next_param = request.args.get('next') or url_for('index')
    return render_template('login.html', next=next_param)

# Logout route
@app.route('/logout')
def logout():
    # mark session end if possible
    try:
        sid = session.get('session_id')
        if sid:
            _record_logout(sid)
    except Exception:
        app.logger.debug('Failed to record logout event')
    session.pop('logged_in', None)
    session.pop('is_admin', None)
    session.pop('session_id', None)
    flash('Déconnecté')
    return redirect(url_for('index'))

# Enforce authentication globally (except login and static files) when enabled
@app.before_request
def enforce_authentication():
    # If auth not enabled, do nothing
    if not AUTH_ENABLED:
        return None
    # Allow login page and static assets and favicon
    path = request.path or ''
    endpoint = request.endpoint or ''
    if endpoint == 'login' or path.startswith('/static') or path == '/favicon.ico':
        return None
    # Allow logout endpoint so users can logout without being redirected
    if endpoint == 'logout':
        return None
    # If user already logged in, allow
    if session.get('logged_in'):
        return None
    # Otherwise redirect to login with next parameter
    return redirect(url_for('login', next=path))

# Inject GitHub repo URL into all templates (optional). Set environment variable GITHUB_REPO_URL to enable.
@app.context_processor
def inject_github_repo():
    try:
        return dict(github_repo_url=os.environ.get('GITHUB_REPO_URL', 'https://github.com/pmourey/luckyIA/tree/copilot/create-ai-module-python'))
    except Exception:
        return dict(github_repo_url='')

@app.context_processor
def inject_auth_state():
    try:
        # Provide a boolean indicating whether the admin endpoint is registered.
        # Avoid calling url_for here (can raise BuildError in some import/test contexts).
        admin_enabled = 'admin_sessions' in app.view_functions
        return dict(auth_enabled=AUTH_ENABLED, logged_in=bool(session.get('logged_in')), admin_enabled=admin_enabled, is_admin=bool(session.get('is_admin')))
    except Exception:
        return dict(auth_enabled=False, logged_in=False, admin_enabled=False, is_admin=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# models folder is at project root 'models'
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# fichier d'historique des conversions
HISTORY_FILE = os.path.join(MODELS_DIR, 'convert_history.json')

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def list_models():
	if not os.path.isdir(MODELS_DIR):
		return []
	# only show common model file extensions
	return sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(('.joblib', '.pkl', '.pth', '.pt', '.onnx'))])


def _reconstruct_torch_module_from_state_dict(state_dict):
    """Tenter de reconstruire un nn.Module minimal à partir d'un state_dict.
    - Repère des clés <prefix>.<idx>.weight (ex: 'network.0.weight') et crée une Sequential
      avec des nn.Linear d'après les shapes des poids.
    - Si ce pattern n'est pas trouvé, tente de créer un unique nn.Linear à partir
      de la première clé se terminant par '.weight'.
    Retourne (module, prefix_name) ou lève RuntimeError si impossible.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        raise RuntimeError('PyTorch requis pour reconstruire un state_dict')

    weight_key_re = re.compile(r"^([a-zA-Z0-9_]+)\.(\d+)\.weight$")
    matches = []
    for k in state_dict.keys():
        m = weight_key_re.match(k)
        if m:
            matches.append((m.group(1), int(m.group(2)), k))

    if matches:
        # choisir le préfixe le plus fréquent
        prefixes = [p for (p, _, _) in matches]
        prefix = Counter(prefixes).most_common(1)[0][0]
        idxs = sorted({idx for (p, idx, _) in matches if p == prefix})
        max_idx = idxs[-1]

        layers = []
        for i in range(max_idx + 1):
            w_key = f"{prefix}.{i}.weight"
            if w_key in state_dict:
                w = state_dict[w_key]
                if hasattr(w, 'shape') and len(w.shape) == 2:
                    in_f = int(w.shape[1])
                    out_f = int(w.shape[0])
                    layers.append(nn.Linear(in_f, out_f))
                else:
                    layers.append(nn.Identity())
            else:
                # pas de poids pour cet index -> activation/dropout probable
                layers.append(nn.Identity())

        seq = nn.Sequential(*layers)
        # Attacher la séquence sur un attribut fixe 'seq' pour permettre scripting
        model_cls = type('ReconstructedModel', (nn.Module,), {})
        model = model_cls()
        setattr(model, 'seq', seq)

        # Remapper les clés du state_dict: prefix.<i>.* -> seq.<i>.* pour loader
        remapped = {}
        prefix_dot = prefix + '.'
        for k, v in state_dict.items():
            if k.startswith(prefix_dot):
                new_k = 'seq.' + k[len(prefix_dot):]
            else:
                new_k = k
            remapped[new_k] = v

        try:
            model.load_state_dict(remapped, strict=False)
            # définir forward qui appelle self.seq
            def forward(self, x):
                return self.seq(x)
            # Attacher forward comme méthode
            setattr(model.__class__, 'forward', forward)
            return model
        except Exception as e:
            raise RuntimeError(f"Échec chargement state_dict reconstruit: {e}")

    # fallback: trouver la première clé '*.weight' et créer un wrapper Linear
    for k in state_dict.keys():
        if k.endswith('.weight'):
            w = state_dict[k]
            if hasattr(w, 'shape') and len(w.shape) == 2:
                in_f = int(w.shape[1])
                out_f = int(w.shape[0])
                class _Wrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = nn.Linear(in_f, out_f)
                    def forward(self, x):
                        return self.linear(x)
                model = _Wrapper()
                try:
                    model.load_state_dict(state_dict, strict=False)
                    return model
                except Exception:
                    raise RuntimeError('Impossible de charger le state_dict dans le wrapper Linear')

    raise RuntimeError('State_dict non reconnu : impossible d_extraire la structure du réseau')


def load_model_file(model_path):
    """Charge un modèle depuis le disque.
    Retourne (model, type) où type est 'sklearn', 'torch' ou 'onnx'.
    Lance une exception si le chargement échoue ou si le format n'est pas supporté.
    """
    lower = model_path.lower()
    if lower.endswith(('.joblib', '.pkl')):
        model = load(model_path)
        return model, 'sklearn'
    elif lower.endswith('.onnx'):
        # Charger via onnxruntime si disponible
        try:
            import onnxruntime as ort
        except Exception:
            raise RuntimeError("onnxruntime n'est pas installé. Installez 'onnxruntime' pour utiliser les fichiers .onnx, ou installez PyTorch et sauvegardez en .pt/.pth")
        try:
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            raise RuntimeError(f"Erreur chargement modèle ONNX: {e}")
        return sess, 'onnx'
    elif lower.endswith(('.pt', '.pth')):
        # Charger via torch si disponible
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise RuntimeError("PyTorch n'est pas installé dans l'environnement. Installez torch pour utiliser les modèles .pt/.pth: https://pytorch.org")

        obj = torch.load(model_path, map_location='cpu')

        # Si on a un dict, tenter d'en extraire un state_dict ou un checkpoint
        if isinstance(obj, dict):
            # Supporter checkpoints sauvegardés par ModelTrainer.save_model
            if 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
                state_dict = obj['model_state_dict']
            else:
                # supposons que l'objet est un state_dict
                state_dict = obj

            # Tenter de reconstruire un module minimal
            try:
                model = _reconstruct_torch_module_from_state_dict(state_dict)
                try:
                    model.eval()
                except Exception:
                    pass
                return model, 'torch'
            except Exception as e:
                # Si reconstruction a échoué, fournir message clair
                raise RuntimeError("Le fichier PyTorch contient un state_dict. Impossible de reconstruire automatiquement le modèle: " + str(e))

        # Cas le plus simple : objet qui est déjà un Module ou ScriptModule
        try:
            if isinstance(obj, torch.nn.Module):
                model = obj
            else:
                model = obj
            try:
                model.eval()
            except Exception:
                pass
            return model, 'torch'
        except Exception as e:
            raise RuntimeError(f"Impossible d'utiliser le fichier PyTorch: {e}")
    else:
        raise RuntimeError("Extension de modèle non supportée. Formats supportés: .joblib, .pkl, .pt, .pth, .onnx")


@app.route('/', methods=['GET'])
def index():
	models = list_models()
	return render_template('index.html', models=models)


@app.route('/predict-page', methods=['GET'])
def predict_page():
    """Page dédiée pour téléverser un CSV et lancer une prédiction (formulaire POST sur /predict)."""
    models = list_models()
    return render_template('predict.html', models=models)


@app.route('/predict', methods=['POST'])
def predict():
	uploaded_file = request.files.get('datafile')
	model_name = request.form.get('model')

	app.logger.debug(f"Received model_name: {model_name}")
	app.logger.debug(f"Received uploaded_file: {uploaded_file}")


	if uploaded_file is None or uploaded_file.filename == '':
		return render_template('results.html', error='Aucun fichier CSV envoyé.')
	if not allowed_file(uploaded_file.filename):
		return render_template('results.html', error='Type de fichier non autorisé. Attendu: .csv')
	if not model_name:
		return render_template('results.html', error='Aucun modèle sélectionné.')

	filename = secure_filename(uploaded_file.filename)

	try:
		df = pd.read_csv(uploaded_file)
	except Exception as e:
		return render_template('results.html', error=f"Erreur lecture CSV: {e}")

	model_path = os.path.join(MODELS_DIR, model_name)
	if not os.path.isfile(model_path):
		return render_template('results.html', error='Modèle introuvable sur le serveur.')

	try:
		model, model_type = load_model_file(model_path)
	except Exception as e:
		return render_template('results.html', error=f"Erreur chargement modèle: {e}")

	try:
		if model_type == 'sklearn':
			preds = model.predict(df)
		elif model_type == 'onnx':
			# Préparer un tableau numpy de floats à partir des colonnes numériques
			numeric_df = df.select_dtypes(include=[np.number])
			if numeric_df.shape[1] == 0:
				return render_template('results.html', error='Le CSV ne contient pas de colonnes numériques nécessaires pour le modèle ONNX.')
			X = numeric_df.to_numpy(dtype=np.float32)

			# Get first input name
			try:
				input_name = model.get_inputs()[0].name
			except Exception:
				input_name = model.get_inputs()[0].name if hasattr(model, 'get_inputs') else None

			inputs = {input_name: X} if input_name is not None else {model.get_inputs()[0].name: X}

			try:
				outs = model.run(None, inputs)
			except Exception as e:
				return render_template('results.html', error=f"Erreur lors de l'inférence ONNX: {e}")

			# Normaliser la sortie
			preds = np.array(outs[0])
			if preds.ndim > 1 and preds.shape[1] == 1:
				preds = preds.ravel()
		elif model_type == 'sklearn':
			preds = model.predict(df)
		elif model_type == 'torch':
			# Préparer un tableau numpy de floats à partir des colonnes numériques
			numeric_df = df.select_dtypes(include=[np.number])
			if numeric_df.shape[1] == 0:
				return render_template('results.html', error='Le CSV ne contient pas de colonnes numériques nécessaires pour le modèle PyTorch.')
			X = numeric_df.to_numpy(dtype=np.float32)

			# --- BLOCK MODIFIÉ ---
			# Préparer le tenseur en fonction du modèle
			try:
				import torch

				tensor, error_msg, warning_msg = _prepare_tensor_for_model(df, model)
				if tensor is None:
					return render_template('results.html', error=error_msg)
				# Si un warning a été généré (ex: suppression automatique de la colonne cible), le remonter aussi via flash
				if warning_msg:
					try:
						flash(warning_msg)
						# parse any removed column names from the warning and persist into model sidecar
						try:
							detected = _parse_removed_columns_from_warning(warning_msg)
							_save_detected_target_in_meta(model_name, detected)
						except Exception:
							app.logger.debug('Failed to persist detected target_column from warning')
					except Exception:
						# Ne pas planter la requête si flash échoue pour une raison quelconque
						app.logger.debug('Failed to flash warning message')
			except Exception as e:
				return render_template('results.html', error=f"Erreur lors de la préparation du tenseur: {e}")
			# --- FIN DU BLOCK MODIFIÉ ---

			try:
				model_device = next(model.parameters()).device if hasattr(model, 'parameters') else None
			except Exception:
				model_device = None

			# Assurer l'utilisation du CPU
			try:
				tensor = tensor.to('cpu')
			except Exception:
				pass

			try:
				model.to('cpu')
			except Exception:
				pass

			import torch

			with torch.no_grad():
				out = model(tensor)

			# out peut être un Tensor ou un tuple; on tente de normaliser en numpy 1D
			if isinstance(out, torch.Tensor):
				preds = out.cpu().numpy()
				if preds.ndim > 1 and preds.shape[1] == 1:
					preds = preds.ravel()
			elif isinstance(out, (list, tuple, np.ndarray)):
				preds = np.array(out)
			else:
				# Fallback : essayer de convertir
				try:
					preds = np.array(out)
				except Exception as e:
					return render_template('results.html', error=f"Impossible d'interpréter la sortie du modèle PyTorch: {e}")
		else:
			return render_template('results.html', error='Type de modèle inconnu.')
	except Exception as e:
		return render_template('results.html', error=f"Erreur lors de la prédiction: {e}")

	# Afficher warning si fourni
	# Construire un HTML d'aperçu des résultats
	table_html = ''
	try:
		import pandas as _pd
		# normaliser preds en numpy 1D ou 2D
		arr = np.array(preds)
		if arr.ndim == 0:
			# scalar
			res_df = _pd.DataFrame({'prediction': [arr.item()] * len(df)})
		elif arr.ndim == 1:
			if arr.shape[0] == len(df):
				res_df = df.copy()
				res_df['prediction'] = arr
			else:
				# mismatch length: show predictions separately
				res_df = _pd.DataFrame({'prediction': arr})
		else:
			# multi-dim predictions: create columns pred_0..pred_n
			cols = {f'pred_{i}': arr[:, i] for i in range(arr.shape[1])}
			if arr.shape[0] == len(df):
				res_df = df.copy()
				for k, v in cols.items():
					res_df[k] = v
			else:
				res_df = _pd.DataFrame(cols)
		# limit rows for preview
		table_html = res_df.head(200).to_html(index=False, classes='table table-sm table-striped')
	except Exception as e:
		# fallback: show plain text message
		table_html = f'<pre>Impossible de générer le tableau de résultats: {e}</pre>'

	return render_template('results.html', table_html=table_html, model=model_name, warning=warning_msg if 'warning_msg' in locals() else None)


@app.route('/api/predict', methods=['POST'])
def api_predict():
	"""Endpoint JSON pour prédire.
	Attendu JSON: {"model": "nom_fichier", "data": [...], "columns": [optional]}
	- data: list of dicts (records) OR list of lists (rows) with provided columns
	"""
	if not request.is_json:
		return jsonify(error='Expected application/json body'), 400
	body = request.get_json()
	model_name = body.get('model')
	data = body.get('data')
	columns = body.get('columns')

	if not model_name or data is None:
		return jsonify(error='Both "model" and "data" are required in the JSON body'), 400

	# sanitize model name to avoid path traversal
	model_name = os.path.basename(model_name)
	model_path = os.path.join(MODELS_DIR, model_name)
	if not os.path.isfile(model_path):
		return jsonify(error='Modèle introuvable'), 404

	try:
		model, model_type = load_model_file(model_path)
	except Exception as e:
		return jsonify(error=f'Erreur chargement modèle: {e}'), 400

	# Construire DataFrame selon le format envoyé
	try:
		if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
			df = pd.DataFrame(data)
		elif isinstance(data, list) and (len(data) == 0 or isinstance(data[0], list)):
			if not columns:
				return jsonify(error='When data is a list of lists, provide "columns" with column names'), 400
			df = pd.DataFrame(data, columns=columns)
		else:
			return jsonify(error='Unsupported data format. Provide a list of dicts or list of lists.'), 400
	except Exception as e:
		return jsonify(error=f'Error building DataFrame: {e}'), 400

	# Faire la prédiction (similaire à /predict)
	try:
		if model_type == 'sklearn':
			preds = model.predict(df)
		elif model_type == 'onnx':
			numeric_df = df.select_dtypes(include=[np.number])
			if numeric_df.shape[1] == 0:
				return jsonify(error='Le DataFrame ne contient pas de colonnes numériques nécessaires pour le modèle ONNX'), 400
			X = numeric_df.to_numpy(dtype=np.float32)
			input_name = model.get_inputs()[0].name
			inputs = {input_name: X}
			outs = model.run(None, inputs)
			preds = np.array(outs[0])
			if preds.ndim > 1 and preds.shape[1] == 1:
				preds = preds.ravel()
		elif model_type == 'torch':
			numeric_df = df.select_dtypes(include=[np.number])
			if numeric_df.shape[1] == 0:
				return jsonify(error='Le DataFrame ne contient pas de colonnes numériques nécessaires pour le modèle PyTorch'), 400
			try:
				import torch
				tensor, error_msg, warning_msg = _prepare_tensor_for_model(df, model)
				if tensor is None:
					return jsonify(error=error_msg), 400
				# Si un warning a été généré (ex: suppression automatique de la colonne cible), le remonter aussi via flash
				if warning_msg:
					try:
						flash(warning_msg)
						# save detected target column(s) to model meta when possible
						try:
							detected = _parse_removed_columns_from_warning(warning_msg)
							_save_detected_target_in_meta(model_name, detected)
						except Exception:
							app.logger.debug('Failed to persist detected target_column from warning (api_predict)')
					except Exception:
						# Ne pas planter la requête si flash échoue pour une raison quelconque
						app.logger.debug('Failed to flash warning message')
			except Exception as e:
				return jsonify(error=f'Erreur lors de la préparation du tenseur: {e}'), 500
			try:
				model.to('cpu')
			except Exception:
				pass
			import torch
			with torch.no_grad():
				out = model(tensor)
			if isinstance(out, torch.Tensor):
				preds = out.cpu().numpy()
				if preds.ndim > 1 and preds.shape[1] == 1:
					preds = preds.ravel()
			else:
				preds = np.array(out)
		else:
			return jsonify(error='Type de modèle inconnu'), 400
	except Exception as e:
		return jsonify(error=f'Erreur lors de la prédiction: {e}'), 400

	preds_list = np.array(preds).tolist()
	preview = df.head(5).to_dict(orient='records')
	resp = {'success': True, 'model': model_name, 'predictions': preds_list, 'preview': preview}
	if 'warning_msg' in locals() and warning_msg:
		resp['warning'] = warning_msg
	return jsonify(resp), 200


@app.route('/generate-data', methods=['GET', 'POST'])
def generate_data():
	"""Générer des données d'exemple."""
	if request.method == 'POST':
		# Récupérer les paramètres du formulaire
		n_samples = request.form.get('n_samples', type=int) or 1000
		random_seed = request.form.get('random_seed', type=int) or 42

		# Générer les données avec les utilitaires d'exemples
		try:
			if ai_examples is None:
				raise RuntimeError("Module ai_module non disponible.")

			# Appeler la fonction de génération
			csv_path = ai_examples.generate_sample_data(output_path=None, n_samples=n_samples, random_state=random_seed)
			# csv_path est un Path ou string
			# Calculer le chemin relatif sous le dossier data/ (ex: raw/client_data.csv)
			try:
				from pathlib import Path
				project_data_dir = Path(PROJECT_ROOT) / 'data'
				csv_path = Path(csv_path)
				# chemin relatif par rapport à data/ (ex: raw/client_data.csv)
				csv_rel = str(csv_path.relative_to(project_data_dir))
			except Exception:
				# fallback: transmettre le nom de fichier
				csv_rel = os.path.basename(str(csv_path))

			return render_template('generate_data.html', success=True, csv_path=str(csv_path), csv_rel=csv_rel)

		except Exception as e:
			return render_template('generate_data.html', error=f"Erreur lors de la génération des données: {e}")

	return render_template('generate_data.html')


@app.route('/download-data/<path:filename>', methods=['GET'])
def download_data(filename):
    """Permet à l'utilisateur de télécharger un fichier à partir du dossier data/ du projet.
    Le paramètre filename est relatif à PROJECT_ROOT/data/ (ex: 'raw/client_data.csv').
    """
    try:
        # sanitize
        safe = os.path.normpath(filename)
        # Prevent path traversal outside data dir
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        full = os.path.join(data_dir, safe)
        if not os.path.isfile(full):
            return Response('Fichier non trouvé', status=404)
        # send as attachment
        return send_from_directory(data_dir, safe, as_attachment=True)
    except Exception as e:
        return Response(str(e), status=500)


@app.route('/generate-model', methods=['GET', 'POST'])
def generate_model():
	"""Générer un ensemble de modèles d'exemple (sklearn + PyTorch optionnel)."""
	if request.method == 'POST':
		try:
			if ai_examples is None:
				raise RuntimeError("Module ai_module non disponible.")

			results = ai_examples.generate_demo_models()
			return render_template('generate_model.html', success=True, results=results)
		except Exception as e:
			return render_template('generate_model.html', error=f"Erreur lors de la génération du modèle: {e}")

	return render_template('generate_model.html')


@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
	"""Entraîner un modèle sur des données téléchargées via l'UI.

	Le CSV doit contenir la colonne cible (par défaut 'risque' ou 'target').
	"""
	if request.method == 'POST':
		datafile = request.files.get('datafile')
		target_col = request.form.get('target_col') or 'risque'
		features = request.form.get('features')  # optional comma-separated
		epochs = request.form.get('epochs', type=int) or 30

		if datafile is None or datafile.filename == '':
			return render_template('train_model.html', error='Aucun fichier de données envoyé.')
		if not allowed_file(datafile.filename):
			return render_template('train_model.html', error='Type de fichier non autorisé. Attendu: .csv')

		# Sauvegarder le fichier de données
		try:
			filename = secure_filename(datafile.filename)
			data_path = os.path.join(BASE_DIR, 'data', filename)
			os.makedirs(os.path.dirname(data_path), exist_ok=True)
			datafile.save(data_path)
		except Exception as e:
			return render_template('train_model.html', error=f"Erreur lors de l'enregistrement du fichier de données: {e}")

		# Préparer feature_columns si fourni
		feature_cols = None
		if features:
			feature_cols = [c.strip() for c in features.split(',') if c.strip()]

		# Lancer l'entraînement via ai_examples
		try:
			if ai_examples is None:
				raise RuntimeError("Module ai_module non disponible.")

			model_output = os.path.join(PROJECT_ROOT, 'models', f"trained_{secure_filename(filename.split('.')[0])}.pth")

			res = ai_examples.train_model_from_csv(data_path=data_path, target_column=target_col, feature_columns=feature_cols, model_output_path=model_output, epochs=epochs)

			return render_template('train_model.html', success=True, model_path=res.get('model_path'), history=res.get('history'))

		except Exception as e:
			return render_template('train_model.html', error=f"Erreur lors de l'entraînement du modèle: {e}")

	# GET
	models = list_models()
	return render_template('train_model.html', models=models)


def _append_conversion_history(entry: dict):
	try:
		os.makedirs(MODELS_DIR, exist_ok=True)
		if os.path.isfile(HISTORY_FILE):
			with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
				data = json.load(f)
		else:
			data = []
	except Exception:
		data = []
	# normalize timestamp if not present
	if 'timestamp' not in entry:
		entry['timestamp'] = datetime.now(timezone.utc).isoformat()
	data.append(entry)
	try:
		with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
			json.dump(data, f, indent=2, ensure_ascii=False)
	except Exception as e:
		app.logger.error(f"Failed to write history file: {e}")


def ensure_meta_files():
    """Inspecte les fichiers du dossier MODELS_DIR et crée des sidecars `<basename>.meta.json` pour ceux qui en sont dépourvus.
    Méthode sûre: toutes les importations lourdes sont encapsulées dans des try/except pour éviter d'échouer si un package manque.
    """
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception:
        return

    for f in list_models():
        base = os.path.splitext(f)[0]
        meta_path = os.path.join(MODELS_DIR, base + '.meta.json')
        if os.path.isfile(meta_path):
            continue
        meta = {
            'created': datetime.now(timezone.utc).isoformat(),
            'source': f,
            'output': None,
            'method': 'inspected',
            'expected_input_dim': None,
            'feature_names': None,
            'note': None
        }
        full = os.path.join(MODELS_DIR, f)
        try:
            lower = f.lower()
            if lower.endswith(('.joblib', '.pkl')):
                try:
                    from joblib import load as jl_load
                    m = jl_load(full)
                    fn = getattr(m, 'feature_names_in_', None)
                    if fn is not None:
                        meta['feature_names'] = list(fn)
                        meta['expected_input_dim'] = len(fn)
                    else:
                        n = getattr(m, 'n_features_in_', None)
                        meta['expected_input_dim'] = int(n) if n is not None else None
                    meta['method'] = 'sklearn_inspect'
                except Exception as e:
                    meta['note'] = f'sklearn_inspect_error: {e}'
            elif lower.endswith('.onnx'):
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(full, providers=['CPUExecutionProvider'])
                    inp = sess.get_inputs()[0]
                    shape = inp.shape
                    if len(shape) >= 2:
                        meta['expected_input_dim'] = int(shape[1]) if shape[1] not in (None, 'None') else None
                    elif len(shape) == 1:
                        meta['expected_input_dim'] = int(shape[0]) if shape[0] not in (None, 'None') else None
                    meta['method'] = 'onnx_inspect'
                except Exception as e:
                    meta['note'] = f'onnx_inspect_error: {e}'
            elif lower.endswith(('.pt', '.pth')):
                try:
                    import torch
                    obj = torch.load(full, map_location='cpu')
                    if isinstance(obj, dict):
                        meta['method'] = 'torch_state_dict_inspect'
                        try:
                            model = _reconstruct_torch_module_from_state_dict(obj.get('model_state_dict', obj))
                            meta['expected_input_dim'] = _get_torch_model_input_dim(model)
                        except Exception as e:
                            meta['note'] = f'reconstruct_error: {e}'
                    elif isinstance(obj, torch.nn.Module):
                        meta['method'] = 'torch_module_inspect'
                        meta['expected_input_dim'] = _get_torch_model_input_dim(obj)
                    else:
                        meta['method'] = 'torch_unknown'
                except Exception as e:
                    meta['note'] = f'torch_inspect_error: {e}'
            else:
                meta['note'] = 'unsupported_ext'
        except Exception as e:
            meta['note'] = f'inspect_error: {e}'
        try:
            _write_meta_file(base, meta)
        except Exception:
            app.logger.debug(f"Failed to write meta for {f}")


def get_models_info():
    """Retourne une liste d'objets décrivant les modèles présents dans MODELS_DIR.
    Chaque item contient: name, path, kind, note, expected_input_dim, feature_names
    Lit en priorité un sidecar JSON `<basename>.meta.json` si présent.
    """
    # assurer la génération des sidecars manquants avant l'inspection
    try:
        ensure_meta_files()
    except Exception:
        app.logger.debug('ensure_meta_files failed')

    files = list_models()
    info = []
    for f in files:
        path = os.path.join(MODELS_DIR, f)
        kind = 'unknown'
        note = None
        expected_input_dim = None
        feature_names = None

        # tenter de lire un sidecar meta (prioritaire)
        try:
            meta_path = os.path.join(MODELS_DIR, f"{os.path.splitext(f)[0]}.meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                # charger infos depuis le sidecar
                expected_input_dim = meta.get('expected_input_dim')
                feature_names = meta.get('feature_names')
                # si le sidecar contient un type/method, on peut noter cela
                note = meta.get('note') if meta.get('note') else None
                # essayer d'inférer kind succinctement
                if meta.get('method') in ('scripted', 'torch_save_fallback'):
                    kind = 'torch_module'
                else:
                    kind = 'unknown'
                info.append({
                    'name': f,
                    'path': path,
                    'kind': kind,
                    'note': note,
                    'expected_input_dim': expected_input_dim,
                    'feature_names': feature_names
                })
                continue
        except Exception as e:
            # si la lecture du sidecar échoue, continuer avec inspection normale
            app.logger.debug(f"Failed to read meta sidecar for {f}: {e}")

        try:
            lower = f.lower()
            if lower.endswith(('.joblib', '.pkl')):
                kind = 'sklearn'
                try:
                    from joblib import load as jl_load
                    m = jl_load(path)
                    # scikit-learn stores feature names in attribute feature_names_in_ (>=0.24)
                    feature_names = getattr(m, 'feature_names_in_', None)
                    if feature_names is None:
                        # try n_features_in_
                        n = getattr(m, 'n_features_in_', None)
                        expected_input_dim = int(n) if n is not None else None
                    else:
                        feature_names = list(feature_names)
                        expected_input_dim = len(feature_names)
                except Exception as e:
                    note = str(e)
            elif lower.endswith('.onnx'):
                kind = 'onnx'
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    inp = sess.get_inputs()[0]
                    shape = inp.shape
                    # heuristique: shape like [None, N] or [N]
                    if len(shape) >= 2:
                        expected_input_dim = int(shape[1]) if shape[1] not in (None, 'None') else None
                    elif len(shape) == 1:
                        expected_input_dim = int(shape[0]) if shape[0] not in (None, 'None') else None
                except Exception as e:
                    note = str(e)
            elif lower.endswith(('.pt', '.pth')):
                # inspect torch objects
                try:
                    import torch
                    obj = torch.load(path, map_location='cpu')
                    if isinstance(obj, dict):
                        kind = 'torch_state_dict'
                        # attempt reconstruct and detect expected dim
                        try:
                            model = _reconstruct_torch_module_from_state_dict(obj.get('model_state_dict', obj))
                            expected_input_dim = _get_torch_model_input_dim(model)
                        except Exception as e:
                            note = f"reconstruct_err: {e}"
                    elif isinstance(obj, torch.nn.Module):
                        kind = 'torch_module'
                        expected_input_dim = _get_torch_model_input_dim(obj)
                    else:
                        kind = 'torch_object'
                except Exception as e:
                    kind = 'torch_unknown'
                    note = str(e)
            else:
                kind = 'other'
        except Exception as e:
            kind = 'error'
            note = str(e)
        info.append({
            'name': f,
            'path': path,
            'kind': kind,
            'note': note,
            'expected_input_dim': expected_input_dim,
            'feature_names': feature_names
        })
    # load history if exists
    history = []
    try:
        if os.path.isfile(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as hf:
                history = json.load(hf)
    except Exception:
        history = []
    return info, history


@app.route('/models', methods=['GET'])
def models_list():
    info, history = get_models_info()
    return render_template('models.html', models=info, history=history)


def convert_state_dict_file(model_filename):
    """Convertit un fichier models/<model_filename> (state_dict) en module scripté .pt.
    Retourne le chemin du fichier scripté.
    """
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    try:
        import torch
    except Exception:
        raise RuntimeError('PyTorch n\'est pas installé dans l\'environnement.')

    obj = torch.load(model_path, map_location='cpu')
    # récupérer state_dict
    if isinstance(obj, dict):
        state_dict = obj.get('model_state_dict', obj)
    else:
        raise RuntimeError('Fichier PyTorch ne contient pas de state_dict.')

    app.logger.debug(f"Converting state_dict for {model_filename} - keys count: {len(state_dict)}")

    # reconstruire
    model = _reconstruct_torch_module_from_state_dict(state_dict)

    # essayer de script
    try:
        scripted = torch.jit.script(model)
        out_name = os.path.splitext(model_filename)[0] + '_scripted.pt'
        out_path = os.path.join(MODELS_DIR, out_name)
        torch.jit.save(scripted, out_path)
        app.logger.debug(f"Scripted model saved to {out_path}")
        # enregistrer historique
        try:
            _append_conversion_history({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': model_filename,
                'output': os.path.basename(out_path),
                'method': 'scripted',
                'note': None
            })
        except Exception:
            app.logger.debug('Failed to append conversion history')
        # écrire sidecar meta
        try:
            expected = None
            try:
                expected = _get_torch_model_input_dim(model)
            except Exception:
                expected = None
            meta = {
                'created': datetime.now(timezone.utc).isoformat(),
                'source': model_filename,
                'output': os.path.basename(out_path),
                'method': 'scripted',
                'expected_input_dim': expected,
                'feature_names': None
            }
            _write_meta_file(os.path.splitext(out_name)[0], meta)
        except Exception:
            app.logger.debug('Failed to write meta sidecar')
        return out_path
    except Exception as e:
        app.logger.debug(f"Scripting failed: {e}; attempting torch.save fallback for {model_filename}")
        # fallback : sauvegarder le module complet (peut ne pas être portable)
        try:
            out_name = os.path.splitext(model_filename)[0] + '_module.pt'
            out_path = os.path.join(MODELS_DIR, out_name)
            torch.save(model, out_path)
            app.logger.debug(f"Fallback module saved to {out_path}")
            # enregistrer historique
            try:
                _append_conversion_history({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': model_filename,
                    'output': os.path.basename(out_path),
                    'method': 'torch_save_fallback',
                    'note': str(e)
                })
            except Exception:
                app.logger.debug('Failed to append conversion history')
            # écrire sidecar meta pour fallback
            try:
                expected = None
                try:
                    expected = _get_torch_model_input_dim(model)
                except Exception:
                    expected = None
                meta = {
                    'created': datetime.now(timezone.utc).isoformat(),
                    'source': model_filename,
                    'output': os.path.basename(out_path),
                    'method': 'torch_save_fallback',
                    'expected_input_dim': expected,
                    'feature_names': None,
                    'note': str(e)
                }
                _write_meta_file(os.path.splitext(out_name)[0], meta)
            except Exception:
                app.logger.debug('Failed to write meta sidecar for fallback')
            return out_path
        except Exception as e2:
            app.logger.error(f"Conversion failed for {model_filename}: {e}; fallback failed: {e2}")
            raise RuntimeError(f"Échec conversion scriptée: {e}; fallback failed: {e2}")


@app.route('/convert-model', methods=['POST'])
def convert_model_route():
    model_name = request.form.get('model') or (request.json and request.json.get('model'))
    if not model_name:
        return jsonify({'success': False, 'error': 'Paramètre model requis'}), 400
    try:
        out_path = convert_state_dict_file(model_name)
        # Si la requête provient d'un formulaire (UI), utiliser flash et redirect
        if request.form:
            from flask import flash
            flash(f"Modèle converti: {model_name} → {os.path.basename(out_path)}")
            return redirect(url_for('models_list'))
        # Sinon retourner JSON
        return jsonify({'success': True, 'converted_path': out_path}), 200
    except Exception as e:
        app.logger.error(f"Conversion error for {model_name}: {e}")
        if request.form:
            flash(f"Erreur conversion: {e}")
            return redirect(url_for('models_list'))
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/models/download/<path:filename>', methods=['GET'])
def download_model(filename):
    """Télécharger un fichier depuis le dossier models."""
    safe_name = os.path.basename(filename)
    return send_from_directory(MODELS_DIR, safe_name, as_attachment=True)


def _get_torch_model_input_dim(model):
    """Essayer d'extraire la dimension d'entrée attendue d'un modèle PyTorch.
    Retourne un int ou None si introuvable.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    # Chercher la première couche nn.Linear dans les modules
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    # Sinon essayer via les paramètres (premier tensor de poids)
    params = list(model.parameters())
    if params:
        w = params[0]
        if w.ndim >= 2:
            # w shape (out, in)
            return int(w.shape[1])
    return None


def _prepare_tensor_for_model(df, model):
    """Prépare un tensor à partir d'un DataFrame pandas en essayant d'aligner
    le nombre de features avec ce que le modèle attend. Retourne (tensor, error_msg, warning_msg).
    Si error_msg est None, tensor est prêt. warning_msg peut contenir une information non bloquante.
    """
    import numpy as _np
    try:
        import torch
    except Exception:
        return None, 'PyTorch requis pour préparer les tenseurs.', None

    numeric_df = df.select_dtypes(include=[_np.number]).copy()
    found = numeric_df.shape[1]
    expected = _get_torch_model_input_dim(model)

    app.logger.debug(f"Preparing tensor: found_numeric_cols={found}, expected_input_dim={expected}")

    warning = None

    # Si on ne connait pas expected, on se contente d'utiliser numeric_df
    if expected is None:
        try:
            tensor = torch.from_numpy(numeric_df.to_numpy(dtype=_np.float32))
            return tensor, None, None
        except Exception as e:
            return None, f"Erreur conversion en tenseur: {e}", None

    # Retirer colonnes cibles ou index fréquents (case-insensitive)
    candidates = ['risque', 'target', 'label', 'unnamed: 0', 'index']
    cols_lower = {c.lower(): c for c in numeric_df.columns}
    removed = []
    for cand in candidates:
        if cand in cols_lower and numeric_df.shape[1] > expected:
            numeric_df.drop(columns=[cols_lower[cand]], inplace=True)
            removed.append(cols_lower[cand])

    found = numeric_df.shape[1]
    app.logger.debug(f"After removing candidates: found={found}, removed={removed}")

    # Si après suppression on a exactement expected -> ok
    if found == expected:
        try:
            tensor = torch.from_numpy(numeric_df.to_numpy(dtype=_np.float32))
            if removed:
                warning = f"Colonnes supprimées automatiquement: {removed}"
                app.logger.debug(warning)
            return tensor, None, warning
        except Exception as e:
            return None, f"Erreur conversion en tenseur: {e}", None

    # Si il y a encore plus de colonnes, auto-trim mais prévenir
    if found > expected:
        cols_kept = list(numeric_df.columns)[:expected]
        trimmed_cols = list(set(numeric_df.columns) - set(cols_kept))
        numeric_df = numeric_df[cols_kept]
        warning = f"Les colonnes suivantes ont été ignorées pour respecter la dimension d'entrée du modèle ({expected}): {trimmed_cols}"
        app.logger.debug(f"Auto-trim columns: kept={cols_kept}, trimmed={trimmed_cols}")
        try:
            tensor = torch.from_numpy(numeric_df.to_numpy(dtype=_np.float32))
            return tensor, None, warning
        except Exception as e:
            return None, f"Erreur conversion en tenseur après trimming: {e}", None

    # Si moins de colonnes que attendu, message d'erreur clair
    if found < expected:
        msg = (
            f"Nombre de colonnes numériques trouvé: {found}, mais le modèle attend {expected}.\n"
            "Solutions possibles: 1) enlever la colonne cible (p.ex. 'risque') du CSV; "
            "2) fournir uniquement les colonnes features attendues; 3) reconvertir/transformer vos données pour correspondre au modèle."
        )
        app.logger.debug(msg)
        return None, msg, None

    # Fallback
    return None, 'Erreur inconnue préparation tenseur', None


def _write_meta_file(model_basename_noext, meta: dict):
    """Écrit un sidecar JSON pour un modèle dans MODELS_DIR: <basename>.meta.json"""
    try:
        meta_path = os.path.join(MODELS_DIR, f"{model_basename_noext}.meta.json")
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2, ensure_ascii=False)
        app.logger.debug(f"Wrote meta file: {meta_path}")
    except Exception as e:
        app.logger.error(f"Failed to write meta file for {model_basename_noext}: {e}")


@app.route('/model-meta/<path:filename>', methods=['GET'])
def model_meta(filename):
    """Retourne le contenu du sidecar meta JSON pour le modèle demandé (ou une inspection si absent)."""
    safe = os.path.basename(filename)
    meta_path = os.path.join(MODELS_DIR, os.path.splitext(safe)[0] + '.meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                data = json.load(mf)
            return jsonify(data), 200
        except Exception as e:
            return Response(str(e), status=500)
    # fallback: inspect quickly
    try:
        info, _ = get_models_info()
        for m in info:
            if m['name'] == safe:
                return jsonify({'source': m['path'], 'method': m['kind'], 'expected_input_dim': m.get('expected_input_dim'), 'feature_names': m.get('feature_names'), 'note': m.get('note')}), 200
        return Response('Model not found', status=404)
    except Exception as e:
        return Response(str(e), status=500)


@app.route('/models/save-meta', methods=['POST'])
def save_model_meta():
    """Sauvegarde le mapping envoyé depuis le formulaire (fields feature_0..N) dans le sidecar meta JSON."""
    model_name = request.form.get('model_name')
    if not model_name:
        return render_template('models.html', error='model_name manquant'), 400
    safe = os.path.basename(model_name)
    # collect feature_* fields
    feature_keys = sorted([k for k in request.form.keys() if k.startswith('feature_')], key=lambda x: int(x.split('_')[1]))
    features = [request.form.get(k) for k in feature_keys]
    # Charger meta existant si présent
    meta = {}
    meta_path = os.path.join(MODELS_DIR, os.path.splitext(safe)[0] + '.meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
        except Exception:
            meta = {}
    # mettre à jour meta
    meta['modified'] = datetime.now(timezone.utc).isoformat()
    meta['feature_names'] = features
    meta['expected_input_dim'] = len(features)
    # write back
    try:
        _write_meta_file(os.path.splitext(safe)[0], meta)
    except Exception as e:
        return render_template('models.html', error=f'Echec écriture meta: {e}'), 500
    # redirect back to index with flash
    flash(f'Meta sauvegardé pour {safe}')
    return redirect(url_for('index'))


@app.route('/models/save-meta-ajax', methods=['POST'])
def save_model_meta_ajax():
    """Endpoint JSON pour sauvegarder le mapping meta via AJAX.
    Attendu JSON: {"model_name": "file.pth", "feature_names": [...], "column_to_feature": {csv_col:feature}}
    """
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Expected application/json'}), 400
    body = request.get_json()
    model_name = body.get('model_name')
    if not model_name:
        return jsonify({'success': False, 'error': 'model_name required'}), 400
    safe = os.path.basename(model_name)
    feature_names = body.get('feature_names')
    column_map = body.get('column_to_feature')

    meta_path = os.path.join(MODELS_DIR, os.path.splitext(safe)[0] + '.meta.json')
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
        except Exception:
            meta = {}

    meta['modified'] = datetime.now(timezone.utc).isoformat()
    if feature_names is not None:
        meta['feature_names'] = feature_names
        meta['expected_input_dim'] = len(feature_names)
    if column_map is not None:
        meta['column_to_feature'] = column_map

    try:
        _write_meta_file(os.path.splitext(safe)[0], meta)
        return jsonify({'success': True, 'meta': meta}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _parse_removed_columns_from_warning(warning_msg: str):
	"""Parse a warning message returned by _prepare_tensor_for_model and extract removed column names.
	Returns a list of column names (may be empty).
	Examples of messages handled:
	- "Colonnes supprimées automatiquement: ['risque']"
	- "Les colonnes suivantes ont été ignorées ...: ['colA', 'colB']"
	"""
	if not warning_msg:
		return []
	import re
	m = re.search(r"\[([^\]]+)\]", warning_msg)
	if not m:
		# fallback: extract quoted tokens
		tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", warning_msg)
		cols = [t[0] or t[1] for t in tokens if (t[0] or t[1])]
		return cols
	inner = m.group(1)
	parts = [p.strip().strip("'\"") for p in inner.split(',') if p.strip()]
	return parts


def _save_detected_target_in_meta(model_filename: str, detected_cols: list):
	"""Update the sidecar meta JSON for the given model (basename) to include target_column and timestamp.
	model_filename can be a basename or path; function uses basename and MODELS_DIR.
	If multiple columns detected, stores a list; otherwise stores a single string.
	"""
	try:
		if not detected_cols:
			return
		safe = os.path.basename(model_filename)
		base = os.path.splitext(safe)[0]
		meta_path = os.path.join(MODELS_DIR, f"{base}.meta.json")
		meta = {}
		if os.path.isfile(meta_path):
			try:
				with open(meta_path, 'r', encoding='utf-8') as mf:
					meta = json.load(mf)
			except Exception:
				meta = {}
		# store single string if length 1, else list
		meta['target_column'] = detected_cols[0] if len(detected_cols) == 1 else detected_cols
		try:
			meta['target_detected_at'] = datetime.now(timezone.utc).isoformat()
		except Exception:
			meta['target_detected_at'] = datetime.utcnow().isoformat() + 'Z'
		_write_meta_file(base, meta)
		app.logger.debug(f"Saved detected target_column for {safe}: {meta['target_column']}")
	except Exception as e:
		app.logger.debug(f"Failed to save detected target in meta for {model_filename}: {e}")


if __name__ == '__main__':
	# logging.basicConfig(level=logging.DEBUG)
	app.run(host='0.0.0.0', port=7000, debug=True)

# Sessions logging utilities
def _sessions_file_path():
    try:
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'auth_sessions.json')
    except Exception:
        return os.path.join(PROJECT_ROOT, 'auth_sessions.json')


def _load_sessions():
    path = _sessions_file_path()
    try:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        app.logger.debug('Failed to load sessions file')
    return []


def _write_sessions(sessions):
    path = _sessions_file_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        app.logger.error(f'Failed to write sessions file: {e}')


def _parse_user_agent(ua_string: str):
    """Try to extract browser+version and os+version from a user-agent string.
    Falls back to returning raw UA when parsing library not available.
    """
    if not ua_string:
        return {'browser': None, 'browser_version': None, 'os': None, 'os_version': None, 'raw': ''}

    # Try to use user_agents library if available
    try:
        from user_agents import parse as ua_parse
        ua = ua_parse(ua_string)
        browser = ua.browser.family
        browser_version = '.'.join([str(x) for x in ua.browser.version if x is not None]) if ua.browser.version else None
        os_name = ua.os.family
        os_version = '.'.join([str(x) for x in ua.os.version if x is not None]) if ua.os.version else None
        return {'browser': browser, 'browser_version': browser_version, 'os': os_name, 'os_version': os_version, 'raw': ua_string}
    except Exception:
        pass

    # Lightweight parsing fallback
    b = None; bv = None; osn = None; osv = None
    ua = ua_string
    # browsers
    m = re.search(r'Chrome/([0-9\.]+)', ua)
    if m:
        b = 'Chrome'; bv = m.group(1)
    m = re.search(r'Firefox/([0-9\.]+)', ua)
    if m and not b:
        b = 'Firefox'; bv = m.group(1)
    m = re.search(r'OPR/([0-9\.]+)', ua)
    if m and not b:
        b = 'Opera'; bv = m.group(1)
    m = re.search(r'Edg/([0-9\.]+)', ua)
    if m and not b:
        b = 'Edge'; bv = m.group(1)
    m = re.search(r'Version/([0-9\.]+).*Safari/', ua)
    if m and not b and 'Safari' in ua:
        b = 'Safari'; bv = m.group(1)

    # os
    if 'Windows' in ua:
        osn = 'Windows'
        m = re.search(r'Windows NT ([0-9\.]+)', ua)
        if m: osv = m.group(1)
    elif 'Mac OS X' in ua or 'Macintosh' in ua:
        osn = 'macOS'
        m = re.search(r'Mac OS X ([0-9_\.]+)', ua)
        if m: osv = m.group(1).replace('_', '.')
    elif 'Android' in ua:
        osn = 'Android'
        m = re.search(r'Android ([0-9\.]+)', ua)
        if m: osv = m.group(1)
    elif 'iPhone OS' in ua or 'iPad' in ua:
        osn = 'iOS'
        m = re.search(r'OS ([0-9_]+)', ua)
        if m: osv = m.group(1).replace('_', '.')
    elif 'Linux' in ua:
        osn = 'Linux'

    return {'browser': b, 'browser_version': bv, 'os': osn, 'os_version': osv, 'raw': ua_string}


def _record_login(session_id, ip, user_agent):
    """Append a login session entry and persist."""
    try:
        sessions = _load_sessions()
        parsed = _parse_user_agent(user_agent)
        entry = {
            'session_id': session_id,
            'ip': ip,
            'user_agent': parsed.get('raw'),
            'browser': parsed.get('browser'),
            'browser_version': parsed.get('browser_version'),
            'os': parsed.get('os'),
            'os_version': parsed.get('os_version'),
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'duration_seconds': None
        }
        sessions.append(entry)
        _write_sessions(sessions)
    except Exception as e:
        app.logger.debug(f'Failed to append login session: {e}')


def _record_logout(session_id):
    """Mark session end_time and duration for session_id."""
    try:
        sessions = _load_sessions()
        updated = False
        for s in reversed(sessions):
            if s.get('session_id') == session_id and s.get('end_time') is None:
                end = datetime.now(timezone.utc)
                s['end_time'] = end.isoformat()
                try:
                    start = datetime.fromisoformat(s['start_time'])
                    s['duration_seconds'] = (end - start).total_seconds()
                except Exception:
                    s['duration_seconds'] = None
                updated = True
                break
        if updated:
            _write_sessions(sessions)
    except Exception as e:
        app.logger.debug(f'Failed to record logout: {e}')


# Admin page for sessions
@app.route('/admin/sessions', methods=['GET'])
@login_required
def admin_sessions():
    try:
        sessions = _load_sessions()
    except Exception:
        sessions = []
    # sort by start_time desc
    try:
        sessions = sorted(sessions, key=lambda x: x.get('start_time') or '', reverse=True)
    except Exception:
        pass

    # Pagination & search
    try:
        page = int(request.args.get('page', 1))
    except Exception:
        page = 1
    try:
        per_page = int(request.args.get('per_page', 20))
    except Exception:
        per_page = 20
    # cap per_page
    if per_page <= 0:
        per_page = 20
    if per_page > 200:
        per_page = 200

    q = (request.args.get('q') or '').strip()
    if q:
        q_lower = q.lower()
        def matches(s):
            for field in ('ip', 'browser', 'os', 'user_agent', 'session_id'):
                v = s.get(field)
                if v and q_lower in str(v).lower():
                    return True
            # also search start_time/end_time
            if s.get('start_time') and q_lower in s.get('start_time').lower():
                return True
            if s.get('end_time') and q_lower in s.get('end_time').lower():
                return True
            return False
        sessions = [s for s in sessions if matches(s)]

    total = len(sessions)
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = sessions[start_idx:end_idx]

    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages
    }

    return render_template('admin_sessions.html', sessions=page_items, pagination=pagination, q=q)

@app.route('/debug/session-info', methods=['GET'])
def debug_session_info():
    """Debug endpoint (local-only) returning current session state and relevant env flags.
    Accessible only from localhost (127.0.0.1 / ::1).
    """
    try:
        # restrict to localhost
        addr = request.remote_addr or ''
        if addr not in ('127.0.0.1', '::1', 'localhost'):
            return Response('Forbidden', status=403)
        # build safe summary
        info = {
            'session_keys': list(session.keys()),
            'session': {k: str(v) for k, v in session.items()},
            'AUTH_ENABLED': AUTH_ENABLED,
            'ADMIN_PASSWORD_present': bool(_ADMIN_RAW),
            'AUTH_passwords_count': len(_plain_passwords) + len(_hashed_passwords),
            'registered_admin_endpoint': 'admin_sessions' in app.view_functions
        }
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Compatibility aliases: accept trailing slash and /admin root
@app.route('/admin/sessions/', methods=['GET'])
@login_required
def admin_sessions_slash():
    """Alias that delegates to admin_sessions to tolerate trailing slash."""
    return admin_sessions()

@app.route('/admin', methods=['GET'])
@login_required
def admin_root_redirect():
    """Redirect /admin to the admin sessions page."""
    return redirect(url_for('admin_sessions'))
