"""
GeneratorPlugin: Transforma vectores latentes en datos sintéticos por tick (primer elemento de la ventana).

Interfaz:
- plugin_params: configuración por defecto.
- plugin_debug_vars: variables expuestas en debug.
- __init__(config): carga y valida parámetros, carga el modelo decoder.
- set_params(**kwargs): actualiza parámetros.
- get_debug_info(): devuelve info de debug.
- add_debug_info(info): añade debug info a un diccionario.
- generate(Z, config): genera datos sintéticos unidimensionales por tick.

"""

import numpy as np
from typing import Dict, Any, List, Optional
from tensorflow.keras.models import load_model, Model


class GeneratorPlugin:
    # Parámetros configurables por defecto
    plugin_params = {
        "sequential_model_file": None,  # Ruta al modelo generador secuencial Keras (por ejemplo, VRNN, CLARM decoder)
        "batch_size_inference": 1,   # Típicamente, la generación secuencial se hace una secuencia a la vez para inferencia
        "history_k": 10,             # Número de pasos pasados para la condicionamiento de la historia
        "num_base_features": 6,      # Número esperado de características base a generar (por ejemplo, HLOC + 2 derivadas)
        "num_fundamental_features": 2 # Número esperado de características fundamentales en la condicionamiento de la historia
        # Agrega cualquier otro parámetro que tu modelo secuencial específico pueda necesitar para la inferencia
    }
    # Variables incluidas en el debug
    plugin_debug_vars = ["sequential_model_file", "batch_size_inference", "history_k", "num_base_features", "num_fundamental_features"]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el GeneratorPlugin y carga el modelo generador secuencial.

        :param config: Diccionario con al menos 'sequential_model_file' y opcionalmente 'batch_size_inference', 'history_k', 'num_base_features', 'num_fundamental_features'.
        :raises ValueError: si no se proporciona 'sequential_model_file'.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)

        model_path = self.params.get("sequential_model_file")
        if not model_path:
            raise ValueError("El parámetro 'sequential_model_file' debe especificar la ruta al modelo generador secuencial.")

        # Carga del modelo generador secuencial preentrenado.
        # Se espera que este modelo maneje la lógica autorregresiva,
        # condicionando sobre Z_t, codificaciones de tiempo, fundamentos y historia.
        try:
            self.sequential_model: Model = load_model(model_path, compile=False)
        except Exception as e:
            raise IOError(f"Error al cargar el modelo secuencial desde {model_path}. Detalle del error: {e}")
        
        self.model = self.sequential_model # Mantener la compatibilidad con GANTrainerPlugin si lo usa directamente

        # Puede que quieras verificar las formas de entrada/salida del modelo cargado aquí
        # para asegurar que coinciden con las expectativas para la generación secuencial.
        # Por ejemplo, self.sequential_model.inputs y self.sequential_model.outputs

    def set_params(self, **kwargs):
        """
        Actualiza parámetros del plugin.

        :param kwargs: pares clave-valor para actualizar plugin_params.
        """
        for key, value in kwargs.items():
            if key in self.plugin_params:
                self.params[key] = value
            # Si la ruta del modelo cambia, recarga el modelo
            if key == "sequential_model_file" and self.params[key] != value: # Verifica si el valor realmente cambió
                self.params[key] = value # Actualiza primero
                model_path = self.params.get("sequential_model_file")
                if model_path:
                    try:
                        self.sequential_model = load_model(model_path, compile=False)
                        self.model = self.sequential_model
                    except Exception as e:
                        raise IOError(f"Error al recargar el modelo secuencial desde {model_path}. Detalle del error: {e}")
                else: # Si la nueva ruta es None/o está vacía
                    raise ValueError("No se puede establecer 'sequential_model_file' a una ruta vacía durante set_params.")


    def get_debug_info(self) -> Dict[str, Any]:
        """
        Devuelve diccionario con valores de debug.

        :return: {var: valor} para cada var en plugin_debug_vars.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]):
        """
        Inserta info de debug en el diccionario proporcionado.

        :param debug_info: diccionario destino.
        """
        debug_info.update(self.get_debug_info())

    def generate(self,
                 feeder_outputs_sequence: List[Dict[str, np.ndarray]],
                 sequence_length_T: int,
                 initial_history_base: Optional[np.ndarray] = None,
                 initial_history_fundamentals: Optional[np.ndarray] = None
                ) -> np.ndarray:
        """
        Genera una secuencia de variables base de manera autorregresiva usando el modelo secuencial cargado.

        El `self.sequential_model` cargado es responsable de la lógica central de generación.
        Este método prepara las entradas para él paso a paso y recopila las salidas.

        :param feeder_outputs_sequence: Lista de diccionarios, uno por paso de tiempo. Cada diccionario de
                                        FeederPlugin.generate contiene 'Z', 'time_encodings',
                                        'current_fundamentals'.
        :param sequence_length_T: El número total de pasos de tiempo a generar.
        :param initial_history_base: Historia inicial opcional para variables base.
                                     Forma (history_k, num_base_features).
        :param initial_history_fundamentals: Historia inicial opcional para variables fundamentales.
                                             Forma (history_k, num_fundamental_features).
        :return: Secuencia de variables base generadas, forma (1, sequence_length_T, num_base_features).
        """
        history_k = self.params.get("history_k")
        num_base_features = self.params.get("num_base_features")
        num_fundamental_features = self.params.get("num_fundamental_features")
        # batch_size_inference = self.params.get("batch_size_inference") # Usualmente 1 para este bucle

        if len(feeder_outputs_sequence) != sequence_length_T:
            raise ValueError("La longitud de feeder_outputs_sequence debe coincidir con sequence_length_T.")

        # Inicializar buffers de historia
        current_history_base = np.zeros((history_k, num_base_features))
        if initial_history_base is not None:
            if initial_history_base.shape == current_history_base.shape:
                current_history_base = initial_history_base.copy()
            else:
                raise ValueError(f"Desajuste de forma en initial_history_base. Se esperaba {(history_k, num_base_features)}, pero se obtuvo {initial_history_base.shape}")
        
        current_history_fundamentals = np.zeros((history_k, num_fundamental_features))
        if initial_history_fundamentals is not None:
            if initial_history_fundamentals.shape == current_history_fundamentals.shape:
                current_history_fundamentals = initial_history_fundamentals.copy()
            else:
                raise ValueError(f"Desajuste de forma en initial_history_fundamentals. Se esperaba {(history_k, num_fundamental_features)}, pero se obtuvo {initial_history_fundamentals.shape}")

        generated_sequence_list = []
        
        # Estado interno de RNN para el modelo, si es stateful y se gestiona externamente.
        # La mayoría de las capas RNN de Keras manejan su propio estado si se usa return_state=True durante la construcción del modelo.
        # Si tu modelo es una sola llamada a Model.predict() para toda la secuencia, este bucle cambia.
        # Este bucle asume una generación paso a paso donde el modelo toma entradas para un paso.
        # rnn_state_h = np.zeros((1, self.sequential_model.get_layer('gru_name').units)) # Ejemplo
        # rnn_state_c = np.zeros((1, self.sequential_model.get_layer('lstm_name').units)) # Ejemplo para LSTM

        for t in range(sequence_length_T):
            feeder_step_output = feeder_outputs_sequence[t]
            # Suponiendo que Z, time_encodings, current_fundamentals ya están en forma (1, feature_dim)
            zt = feeder_step_output["Z"] 
            time_enc_t = feeder_step_output["time_encodings"]
            current_funda_t = feeder_step_output["current_fundamentals"]

            # Preparar entradas para el self.sequential_model.
            # La estructura exacta (lista, diccionario) y las formas dependen de cómo se definió self.sequential_model.
            # Ejemplo: model_inputs = [zt, time_enc_t, current_history_base_flat, current_history_funda_flat, rnn_state_h_in]
            
            # Aplanar historia para capas típicas Dense/RNN si la historia se trata como un vector plano
            history_base_input = current_history_base.reshape(1, -1) 
            history_funda_input = current_history_fundamentals.reshape(1, -1)

            # Este es un marcador de posición para cómo se alimentan las entradas a tu modelo secuencial específico
            # Depende en gran medida de la arquitectura de tu modelo (VRNN, CLARM, etc.)
            # Asegúrate de que las formas coincidan con las capas de entrada del modelo.
            model_inputs = { # Ejemplo si el modelo toma un diccionario
                "latent_input": zt, # Forma (1, latent_dim)
                "time_encoding_input": time_enc_t, # Forma (1, num_time_features)
                "base_history_input": history_base_input, # Forma (1, history_k * num_base_features)
                "fundamental_history_input": history_funda_input, # Forma (1, history_k * num_fundamental_features)
                # "rnn_state_h_input": rnn_state_h, # Si se gestiona el estado explícitamente
            }
            # O si toma una lista:
            # model_input_list = [zt, time_enc_t, history_base_input, history_funda_input]

            # Realizar un paso de generación
            # Si el modelo es stateful y devuelve estados:
            # predicted_output_xt_array, rnn_state_h = self.sequential_model.predict(model_inputs)
            # else:
            predicted_output_xt_array = self.sequential_model.predict(model_inputs)
            
            # Asegurarse de que la salida es (1, num_base_features)
            if not (isinstance(predicted_output_xt_array, np.ndarray) and \
                    predicted_output_xt_array.ndim == 2 and \
                    predicted_output_xt_array.shape[0] == 1 and \
                    predicted_output_xt_array.shape[1] == num_base_features):
                raise ValueError(f"La salida del modelo secuencial para un solo paso tiene una forma inesperada: {predicted_output_xt_array.shape}. Se esperaba (1, {num_base_features})")

            predicted_xt = predicted_output_xt_array[0] # Forma (num_base_features,)
            generated_sequence_list.append(predicted_xt)

            # Actualizar buffers de historia
            current_history_base = np.roll(current_history_base, -1, axis=0)
            current_history_base[-1, :] = predicted_xt # Agregar el nuevo paso generado
            
            current_history_fundamentals = np.roll(current_history_fundamentals, -1, axis=0)
            # current_funda_t es (1, num_fundamental_features), así que toma [0]
            current_history_fundamentals[-1, :] = current_funda_t[0] 

        final_generated_sequence = np.array(generated_sequence_list) # Forma (sequence_length_T, num_base_features)
        return np.expand_dims(final_generated_sequence, axis=0) # Retornar como (1, T, num_base_features)

    def update_model(self, new_model: Model):
        """
        Actualiza el modelo del generador. Usado por GANTrainerPlugin después del entrenamiento de GAN.
        """
        if not isinstance(new_model, Model):
            raise TypeError(f"new_model debe ser un modelo Keras, se recibió {type(new_model)}")
        print("GeneratorPlugin: Actualizando sequential_model con una nueva instancia de modelo.")
        self.sequential_model = new_model
        self.model = new_model # Mantener la alias de self.model consistente
