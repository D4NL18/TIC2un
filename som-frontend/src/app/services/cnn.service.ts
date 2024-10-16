import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CnnService {

  private apiUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  trainCNN(): Observable<any> {
    return this.http.post(`${this.apiUrl}/predict`, {});
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/confusion-matrix`, { responseType: 'blob' });
  }

  getAccuracy(): Observable<any> {
    return this.http.get(`${this.apiUrl}/metrics`);
  }
}