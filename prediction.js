// Prediction Model Logic
document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form values
    const age = parseInt(document.getElementById('age').value);
    const sexe = document.querySelector('input[name="sexe"]:checked').value;
    const typeAdmission = document.querySelector('input[name="type_admission"]:checked').value;
    const severite = document.getElementById('severite').value;
    
    // Validate all fields
    if (!severite) {
        alert('Veuillez sélectionner la sévérité du cas');
        return;
    }
    
    // Calculate predicted duration
    const prediction = predictDuration(age, sexe, typeAdmission, severite);
    
    // Display results
    displayResults(prediction, age, sexe, typeAdmission, severite);
});

// Reset form handler
document.getElementById('predictionForm').addEventListener('reset', function() {
    document.getElementById('result').style.display = 'none';
});

function predictDuration(age, sexe, typeAdmission, severite) {
    // Base duration
    let duration = 3.0;
    
    // Age factor - older patients tend to stay longer
    if (age < 30) {
        duration += 0;
    } else if (age < 50) {
        duration += 1;
    } else if (age < 65) {
        duration += 2;
    } else {
        duration += 4;
    }
    
    // Gender factor - slight difference
    if (sexe === 'M') {
        duration += 0.5;
    } else {
        duration += 0.3;
    }
    
    // Admission type factor
    if (typeAdmission === 'urgence') {
        duration += 2.5;
    } else {
        duration += 0.5;
    }
    
    // Severity factor - most important
    if (severite === 'faible') {
        duration -= 1.5;
    } else if (severite === 'moyen') {
        duration += 1;
    } else if (severite === 'eleve') {
        duration += 5;
    }
    
    // Round to nearest 0.5
    duration = Math.round(duration * 2) / 2;
    
    // Ensure minimum of 1 day
    duration = Math.max(1, duration);
    
    // Calculate confidence interval (±20%)
    const margin = duration * 0.2;
    const lowerBound = Math.max(1, Math.round((duration - margin) * 2) / 2);
    const upperBound = Math.round((duration + margin) * 2) / 2;
    
    return {
        duration: duration,
        lowerBound: lowerBound,
        upperBound: upperBound
    };
}

function displayResults(prediction, age, sexe, typeAdmission, severite) {
    // Show result container
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    
    // Update duration
    const durationSpan = document.getElementById('predictedDuration');
    durationSpan.textContent = `${prediction.duration} jour${prediction.duration > 1 ? 's' : ''}`;
    
    // Update confidence interval
    const intervalSpan = document.getElementById('confidenceInterval');
    intervalSpan.textContent = `${prediction.lowerBound} - ${prediction.upperBound} jours`;
    
    // Update factors list
    const factorsList = document.getElementById('factorsList');
    factorsList.innerHTML = '';
    
    const factors = [
        `Âge: ${age} ans`,
        `Sexe: ${sexe === 'M' ? 'Masculin' : 'Féminin'}`,
        `Type d'admission: ${typeAdmission === 'urgence' ? 'Urgence' : 'Programmée'}`,
        `Sévérité: ${getSeveriteText(severite)}`
    ];
    
    factors.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
    });
    
    // Scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function getSeveriteText(severite) {
    const texts = {
        'faible': 'Faible',
        'moyen': 'Moyen',
        'eleve': 'Élevé'
    };
    return texts[severite] || severite;
}

// Add some interactivity to the form
document.getElementById('age').addEventListener('input', function(e) {
    if (e.target.value < 0) {
        e.target.value = 0;
    } else if (e.target.value > 120) {
        e.target.value = 120;
    }
});

// Clear result when form inputs change
const formInputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
formInputs.forEach(input => {
    input.addEventListener('change', function() {
        // Optionally hide results when form changes
        // document.getElementById('result').style.display = 'none';
    });
});
